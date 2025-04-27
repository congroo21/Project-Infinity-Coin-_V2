# src/optimization/feedback_loop.py

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from collections import deque

@dataclass
class PerformanceMetrics:
    """성과 지표"""
    timestamp: datetime
    realized_return: float
    predicted_return: float
    actual_costs: float
    predicted_costs: float
    holding_period: int
    win_rate: float
    sharpe_ratio: float
    strategy_id: str

class FeedbackLoop:
    """실시간 피드백 루프 시스템"""
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.metrics_history = deque(maxlen=1000)
        self.strategy_performance = {}
        self.active_adjustments = set()
        
        # 성과 임계값 설정
        self.thresholds = {
            'min_win_rate': 0.55,
            'min_sharpe': 1.0,
            'max_drawdown': 0.02,
            'min_trades': 100
        }
        
        # 조정 가능한 파라미터 범위
        self.parameter_ranges = {
            'position_size': (0.01, 0.1),
            'holding_period': (10, 300),  # 초 단위
            'entry_threshold': (0.001, 0.01)
        }

    async def start(self):
        """피드백 루프 시스템 시작"""
        logging.info("Starting feedback loop system...")
        while True:
            try:
                # 성과 분석
                metrics = await self._analyze_performance()
                if metrics:
                    self.metrics_history.append(metrics)
                    
                    # 전략 조정 필요성 확인
                    if self._needs_adjustment(metrics):
                        await self._adjust_strategy(metrics)
                
                # 주기적인 클린업
                await self._cleanup_old_data()
                
                await asyncio.sleep(1)  # 1초마다 체크
                
            except Exception as e:
                logging.error(f"Feedback loop error: {e}")
                await asyncio.sleep(5)

    async def _analyze_performance(self) -> Optional[PerformanceMetrics]:
        """실시간 성과 분석"""
        try:
            # 최근 거래 데이터 수집
            recent_trades = await self._get_recent_trades()
            if not recent_trades:
                return None

            # 실현 수익률 계산
            realized_returns = [trade['pnl'] for trade in recent_trades]
            
            # 예측 정확도 분석
            prediction_accuracy = self._analyze_predictions(recent_trades)
            
            # 승률 계산
            win_rate = len([r for r in realized_returns if r > 0]) / len(realized_returns)
            
            # 샤프 비율 계산
            returns = np.array(realized_returns)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 24 * 60)

            return PerformanceMetrics(
                timestamp=datetime.now(),
                realized_return=np.mean(realized_returns),
                predicted_return=prediction_accuracy['predicted_return'],
                actual_costs=prediction_accuracy['actual_costs'],
                predicted_costs=prediction_accuracy['predicted_costs'],
                holding_period=prediction_accuracy['avg_holding_period'],
                win_rate=win_rate,
                sharpe_ratio=sharpe,
                strategy_id=prediction_accuracy['strategy_id']
            )

        except Exception as e:
            logging.error(f"Performance analysis error: {e}")
            return None

    def _needs_adjustment(self, metrics: PerformanceMetrics) -> bool:
        """전략 조정 필요성 판단"""
        # 성과가 임계값 미달인 경우
        if metrics.win_rate < self.thresholds['min_win_rate']:
            return True
            
        if metrics.sharpe_ratio < self.thresholds['min_sharpe']:
            return True
            
        # 예측과 실제 수익률 차이가 큰 경우
        if abs(metrics.realized_return - metrics.predicted_return) > 0.01:
            return True
            
        # 거래비용이 예상보다 높은 경우
        if metrics.actual_costs > metrics.predicted_costs * 1.2:
            return True
            
        return False

    async def _adjust_strategy(self, metrics: PerformanceMetrics):
        """전략 자동 조정"""
        try:
            if metrics.strategy_id in self.active_adjustments:
                return  # 이미 조정 중인 전략은 스킵
                
            self.active_adjustments.add(metrics.strategy_id)
            
            # 승률 기반 포지션 크기 조정
            if metrics.win_rate < self.thresholds['min_win_rate']:
                await self._adjust_position_size(metrics.strategy_id, decrease=True)
            
            # 수익률 기반 진입 임계값 조정
            if metrics.realized_return < metrics.predicted_return * 0.8:
                await self._adjust_entry_threshold(metrics.strategy_id, increase=True)
            
            # 비용 기반 홀딩 기간 조정
            if metrics.actual_costs > metrics.predicted_costs * 1.2:
                await self._adjust_holding_period(metrics.strategy_id, increase=True)
            
            # 조정 후 모니터링을 위해 성과 초기화
            self.strategy_performance[metrics.strategy_id] = {
                'adjustment_time': datetime.now(),
                'base_metrics': metrics
            }
            
            self.active_adjustments.remove(metrics.strategy_id)
            
        except Exception as e:
            logging.error(f"Strategy adjustment error: {e}")
            if metrics.strategy_id in self.active_adjustments:
                self.active_adjustments.remove(metrics.strategy_id)

    async def _adjust_position_size(self, strategy_id: str, decrease: bool = True):
        """포지션 크기 조정"""
        try:
            current_size = await self._get_current_param(strategy_id, 'position_size')
            min_size, max_size = self.parameter_ranges['position_size']
            
            if decrease:
                new_size = max(min_size, current_size * 0.8)
            else:
                new_size = min(max_size, current_size * 1.2)
                
            await self._update_strategy_param(strategy_id, 'position_size', new_size)
            logging.info(f"Adjusted position size for {strategy_id}: {new_size:.4f}")
            
        except Exception as e:
            logging.error(f"Position size adjustment error: {e}")

    async def _adjust_entry_threshold(self, strategy_id: str, increase: bool = True):
        """진입 임계값 조정"""
        try:
            current_threshold = await self._get_current_param(strategy_id, 'entry_threshold')
            min_threshold, max_threshold = self.parameter_ranges['entry_threshold']
            
            if increase:
                new_threshold = min(max_threshold, current_threshold * 1.2)
            else:
                new_threshold = max(min_threshold, current_threshold * 0.8)
                
            await self._update_strategy_param(strategy_id, 'entry_threshold', new_threshold)
            logging.info(f"Adjusted entry threshold for {strategy_id}: {new_threshold:.4f}")
            
        except Exception as e:
            logging.error(f"Entry threshold adjustment error: {e}")

    async def _adjust_holding_period(self, strategy_id: str, increase: bool = True):
        """보유 기간 조정"""
        try:
            current_period = await self._get_current_param(strategy_id, 'holding_period')
            min_period, max_period = self.parameter_ranges['holding_period']
            
            if increase:
                new_period = min(max_period, current_period * 1.2)
            else:
                new_period = max(min_period, current_period * 0.8)
                
            await self._update_strategy_param(strategy_id, 'holding_period', new_period)
            logging.info(f"Adjusted holding period for {strategy_id}: {new_period:.0f}s")
            
        except Exception as e:
            logging.error(f"Holding period adjustment error: {e}")

    async def _get_current_param(self, strategy_id: str, param_name: str) -> float:
        """현재 파라미터 값 조회"""
        # 실제 구현에서는 DB나 설정에서 조회
        return 0.05  # 예시 값

    async def _update_strategy_param(self, strategy_id: str, param_name: str, value: float):
        """전략 파라미터 업데이트"""
        # 실제 구현에서는 DB 업데이트 및 전략 리로드
        pass

    def _analyze_predictions(self, trades: List[Dict]) -> Dict:
        """예측 정확도 분석"""
        # 예시 데이터
        return {
            'predicted_return': 0.001,
            'actual_costs': 0.0002,
            'predicted_costs': 0.0001,
            'avg_holding_period': 60,
            'strategy_id': 'default_strategy'
        }

    async def _cleanup_old_data(self):
        """오래된 데이터 정리"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # 오래된 메트릭스 제거
            while self.metrics_history and self.metrics_history[0].timestamp < cutoff_time:
                self.metrics_history.popleft()
                
            # 오래된 전략 성과 데이터 제거
            old_strategies = [
                strategy_id for strategy_id, data in self.strategy_performance.items()
                if data['adjustment_time'] < cutoff_time
            ]
            
            for strategy_id in old_strategies:
                del self.strategy_performance[strategy_id]
                
        except Exception as e:
            logging.error(f"Data cleanup error: {e}")

    async def _get_recent_trades(self) -> List[Dict]:
        """최근 거래 데이터 조회"""
        # 실제 구현에서는 DB에서 조회
        return []

    def get_status(self) -> Dict:
        """현재 상태 조회"""
        return {
            'active_adjustments': len(self.active_adjustments),
            'metrics_count': len(self.metrics_history),
            'strategies_monitored': len(self.strategy_performance),
            'latest_metrics': self.metrics_history[-1] if self.metrics_history else None
        }