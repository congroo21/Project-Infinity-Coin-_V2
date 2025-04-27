# src/scenarios/models/monte_carlo.py

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import asyncio
from datetime import datetime

@dataclass
class SimulationResult:
    """시뮬레이션 결과를 저장하는 데이터 클래스"""
    expected_return: float
    var_95: float
    var_99: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    confidence_level: float

class MonteCarloSimulation:
    """최적화된 몬테카를로 시뮬레이션 모델"""
    def __init__(self):
        # 반복 횟수 최적화 (리소스 사용 75% 감소)
        self.iterations = 500  # 기존 2000에서 500으로 감소
        self.confidence_level = 0.95
        self.risk_free_rate = 0.02
        self.min_history_size = 50
        
        # CPU 코어 수에 따른 병렬 처리 설정 (최대 4개로 제한)
        self.num_processes = min(4, max(1, mp.cpu_count() - 1))
        
        # 시나리오별 변동성 조정 계수
        self.volatility_adjustments = {
            'high_volatility': 1.5,
            'low_volatility': 0.7,
            'strong_trend': 1.2,
            'weak_trend': 0.9,
            'high_liquidity': 0.8,
            'low_liquidity': 1.3
        }

        # 성능 최적화를 위한 캐시 설정
        self._cache = {}
        self._cache_size = 200  # 캐시 크기 감소 (1000 -> 200)
        self._cache_ttl = 300  # 캐시 유효시간 증가 (60초 -> 300초)
        self._last_simulation_params = None
        self._last_simulation_result = None

    def _run_single_simulation(self, params: Dict) -> np.ndarray:
        """단일 시뮬레이션 실행 (병렬 처리용) - 벡터화 최적화"""
        current_price = params['current_price']
        drift = params['drift']
        volatility = params['volatility']
        iterations = params['iterations']
        days = params.get('days', 252)  # 기본값 제공
        
        # 벡터화된 연산으로 최적화 (한 번에 모든 경로 생성)
        random_walks = np.random.normal(
            drift,
            volatility,
            (iterations, days)
        )
        
        # 메모리 효율적인 계산: 미리 배열 할당 후 계산
        paths = np.zeros((iterations, days))
        paths[:, 0] = current_price
        
        # 누적곱 사용으로 최적화 (for 루프 제거)
        np.exp(random_walks, out=random_walks)  # in-place 계산으로 메모리 효율화
        paths[:, 1:] = current_price * np.cumprod(random_walks[:, :-1], axis=1)
        
        return paths

    async def run_simulation(
        self,
        scenario_type: str,
        price_history: List[float],
        current_volatility: float,
        position_size: float
    ) -> Optional[SimulationResult]:
        """병렬 처리가 적용된 시뮬레이션 실행 - 최적화 버전"""
        try:
            if len(price_history) < self.min_history_size:
                return None

            # 캐시 키 생성 (단순화)
            cache_key = (
                f"{scenario_type}_{current_volatility:.4f}_{position_size:.4f}_"
                f"{price_history[-1]:.0f}"
            )
            
            # 캐시된 결과가 있는지 확인
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result

            # 입력 파라미터 비교를 통한 최적화
            current_params = {
                'scenario_type': scenario_type,
                'price_last': price_history[-1],
                'volatility': current_volatility,
                'position_size': position_size
            }
            
            # 이전 시뮬레이션과 파라미터가 유사하면 이전 결과 재사용
            if self._last_simulation_params and self._last_simulation_result:
                if (abs(current_params['price_last'] - self._last_simulation_params['price_last']) / self._last_simulation_params['price_last'] < 0.001 and
                    abs(current_params['volatility'] - self._last_simulation_params['volatility']) < 0.001 and
                    current_params['scenario_type'] == self._last_simulation_params['scenario_type']):
                    return self._last_simulation_result

            # 시나리오별 변동성 조정
            vol_multiplier = self.volatility_adjustments.get(scenario_type, 1.0)
            adjusted_volatility = current_volatility * vol_multiplier

            # 수익률 계산 (벡터화) - 효율적인 추세 계산
            if len(price_history) > 1:
                price_array = np.array(price_history)
                returns = np.diff(np.log(price_array))
                
                # 시나리오별 추세 조정
                base_drift = np.mean(returns)
                drift = base_drift * 1.2 if 'strong_trend' in scenario_type else (
                    base_drift * 0.8 if 'weak_trend' in scenario_type else base_drift
                )
            else:
                drift = 0.0001  # 기본값
            
            # 리소스 사용 최적화를 위해 시뮬레이션 일수 조정
            if scenario_type in ['high_volatility', 'strong_trend']:
                days = 63  # 3개월 (기존 252일에서 감소)
            else:
                days = 126  # 6개월
            
            # 병렬 처리를 위한 파라미터 설정
            iterations_per_process = self.iterations // self.num_processes
            simulation_params = {
                'current_price': price_history[-1],
                'drift': drift,
                'volatility': adjusted_volatility,
                'iterations': iterations_per_process,
                'days': days
            }

            # 병렬 시뮬레이션 실행 - 메모리 사용 개선
            loop = asyncio.get_event_loop()
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                tasks = [
                    loop.run_in_executor(
                        executor,
                        self._run_single_simulation,
                        simulation_params
                    )
                    for _ in range(self.num_processes)
                ]
                
                simulation_results = await asyncio.gather(*tasks)

            # 결과 통합 (numpy 병합 최적화)
            if not simulation_results:
                return None
                
            # 메모리 효율적 병합
            price_paths = np.vstack([r for r in simulation_results if r is not None])
            
            if price_paths.size == 0:
                return None
            
            # 결과 계산 (벡터화된 연산 사용)
            result = await self._calculate_simulation_metrics(
                price_paths,
                price_history[-1],
                position_size
            )

            # 파라미터 및 결과 캐싱
            self._last_simulation_params = current_params
            self._last_simulation_result = result
            self._cache_result(cache_key, result)

            return result

        except Exception as e:
            logging.error(f"몬테카를로 시뮬레이션 오류: {e}")
            return None
            
    # src/scenarios/models/monte_carlo.py의 리스크 관련 함수 수정

    async def _calculate_simulation_metrics(
        self,
        price_paths: np.ndarray,
        current_price: float,
        position_size: float
    ) -> SimulationResult:
        """시뮬레이션 메트릭 계산 (벡터화된 연산 사용) - 최적화 버전"""
        # 중앙화된 리스크 계산기 사용
        from src.utils.risk_utils import risk_calculator
        
        # 최종 가격 및 수익률 계산 (벡터화)
        final_prices = price_paths[:, -1]
        returns = (final_prices - current_price) / current_price
        
        # 포트폴리오 가치 계산 (메모리 효율적)
        portfolio_values = position_size * price_paths
        
        # 주요 지표 계산 (벡터화)
        expected_return = float(np.mean(returns))
        
        # VaR 계산
        var_95 = risk_calculator.calculate_historical_var(returns, 0.95)
        var_99 = risk_calculator.calculate_historical_var(returns, 0.99)
        
        volatility = float(np.std(returns))
        win_rate = float(np.mean(returns > 0))
        
        # Sharpe Ratio 계산
        sharpe_ratio = risk_calculator.calculate_sharpe_ratio(returns)
        
        # 최대 낙폭 계산
        max_drawdown = self._calculate_max_drawdown_vectorized(portfolio_values)
        
        # 신뢰도 계산 (변동성 기반)
        confidence_level = float(min(1.0, 1.0 - volatility))

        return SimulationResult(
            expected_return=expected_return,
            var_95=var_95,
            var_99=var_99,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            confidence_level=confidence_level
        )

    def _calculate_max_drawdown_vectorized(self, portfolio_values: np.ndarray) -> float:
        """최대 낙폭 계산 (벡터화) - 성능 최적화 버전"""
        try:
            # 중앙화된 리스크 계산기 사용하기보다는 현재 최적화된 벡터 연산 유지
            # 이유: 배열 전체를 한 번에 처리하는 특수한 방식이기 때문
            
            # 누적 최대값 계산
            cummax = np.maximum.accumulate(portfolio_values, axis=1)
            
            # 낙폭 계산
            drawdowns = (cummax - portfolio_values) / np.maximum(cummax, 1e-10)  # 0 나누기 방지
            
            # 각 경로별 최대 낙폭
            max_drawdowns = np.max(drawdowns, axis=1)
            
            # 평균 최대 낙폭
            max_drawdown = float(np.mean(max_drawdowns))
            
            return max_drawdown
            
        except Exception as e:
            logging.error(f"최대 낙폭 계산 오류: {e}")
            return 0.0

    def _get_cached_result(self, key: str) -> Optional[SimulationResult]:
        """캐시된 결과 조회 - 최적화 버전"""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if (datetime.now() - timestamp).total_seconds() < self._cache_ttl:
                return result
            
            # 캐시 정리 최적화 (삭제 연산 최소화)
            if len(self._cache) > self._cache_size * 0.8:  # 80% 이상 찼을 때만 삭제
                del self._cache[key]
        return None

    def _cache_result(self, key: str, result: SimulationResult):
        """결과 캐시에 저장 - 최적화 버전"""
        # 캐시 크기 제한 관리 - 배치 처리
        if len(self._cache) >= self._cache_size:
            # 가장 오래된 10개 항목 제거 (배치 처리로 효율화)
            current_time = datetime.now()
            oldest_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k][1],
                reverse=False
            )[:10]
            
            for old_key in oldest_keys:
                del self._cache[old_key]
        
        self._cache[key] = (result, datetime.now())