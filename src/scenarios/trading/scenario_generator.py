import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ...market_analyzer import MarketState
from ..models.base_models import MarketScenario, AnalysisResult
from ..models.bayesian_model import BayesianModel
from ..models.monte_carlo import MonteCarloSimulation
from ..market.state_manager import MarketStateManager
from ..monitoring.performance import ScenarioPerformanceMonitor
from ...exceptions import (
    ValidationError, 
    InsufficientDataError,
    MarketDataError,
    SystemResourceError
)
from ...config import Config

@dataclass
class ScenarioConfig:
    """시나리오 생성 설정"""
    min_volatility: float = 0.01     # 최소 변동성
    max_volatility: float = 0.05     # 최대 변동성
    trend_threshold: float = 0.02    # 추세 임계값
    liquidity_threshold: float = 0.7  # 유동성 임계값
    confidence_threshold: float = 0.6 # 신뢰도 임계값
    min_data_points: int = 7         # 최소 데이터 포인트
    cache_size: int = 1000           # 캐시 최대 크기
    cleanup_interval: int = 3600     # 정리 주기(초)

@dataclass
class ScenarioContext:
    """시나리오 컨텍스트 데이터"""
    timestamp: datetime
    market_conditions: Dict
    historical_data: Dict
    risk_metrics: Dict
    performance_stats: Optional[Dict] = None

    def validate(self) -> bool:
        """컨텍스트 유효성 검증"""
        try:
            if not isinstance(self.timestamp, datetime):
                return False
            if not isinstance(self.market_conditions, dict):
                return False
            if not isinstance(self.historical_data, dict):
                return False
            if not self.historical_data.get('prices'):
                return False
            if not isinstance(self.risk_metrics, dict):
                return False
            return True
        except Exception as e:
            logging.error(f"Context validation error: {e}")
            return False

class ScenarioGenerator:
    """향상된 시나리오 생성 시스템"""
    def __init__(self, config: Optional[ScenarioConfig] = None):
        # 컴포넌트 초기화
        self.config = config or ScenarioConfig()
        self.market_state = MarketStateManager()
        self.bayesian_model = BayesianModel()
        self.monte_carlo = MonteCarloSimulation()
        self.performance_monitor = ScenarioPerformanceMonitor()
        
        # 시나리오 이력
        self.scenario_history = []
        self.last_scenario = None
        self.scenario_cache = {}
        self.last_cleanup = datetime.now()
        
        # 상태 추적
        self.total_scenarios = 0
        self.successful_scenarios = 0
        self.error_count = 0
        
        # 실행 통계
        self.execution_stats = {
            'execution_time': 0.0,
            'success_rate': 0.0,
            'error_rate': 0.0,
            'avg_latency': 0.0,
            'last_update': None
        }
        
        # 시장 상태 캐시
        self.market_state_cache = {}
        self.last_market_update = None
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """로깅 설정"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    async def _create_scenario_context(self, market_data: Dict) -> Optional[ScenarioContext]:
        """시나리오 컨텍스트 생성"""
        # 시장 상태 업데이트 (잘못된 데이터가 들어올 경우 market_state.update()에서 예외가 발생함)
        self.market_state.update(market_data)
        current_state = self.market_state.get_current_state()
        
        # 히스토리 데이터 수집
        historical_data = await self._collect_historical_data(market_data)
        if not historical_data or len(historical_data.get('prices', [])) < self.config.min_data_points:
            self.logger.warning(
                f"Insufficient data points: {len(historical_data.get('prices', []))} < {self.config.min_data_points}"
            )
            return None

        # 리스크 메트릭스 계산
        risk_metrics = self._calculate_risk_metrics(historical_data)
        if not risk_metrics:
            raise MarketDataError("Failed to calculate risk metrics")
        
        # 성능 통계 수집
        performance_stats = self.performance_monitor.get_performance_metrics()

        context = ScenarioContext(
            timestamp=datetime.now(),
            market_conditions=current_state,
            historical_data=historical_data,
            risk_metrics=risk_metrics,
            performance_stats=performance_stats
        )

        if not context.validate():
            raise ValidationError("Invalid scenario context")

        return context

    async def _collect_historical_data(self, market_data: Dict) -> Dict:
        """히스토리 데이터 수집"""
        try:
            prices = []
            volumes = []
            timestamps = []

            # MarketStateManager의 integrated_history를 활용하여 과거 데이터를 수집
            if hasattr(self.market_state, 'integrated_history'):
                for state in self.market_state.integrated_history:
                    prices.append(state.price)
                    volumes.append(state.volume)
                    timestamps.append(state.timestamp)

            # 혹시 현재 market_data가 integrated_history에 포함되지 않았다면 추가
            current_price = market_data.get('price')
            current_volume = market_data.get('volume')
            current_time = market_data.get('timestamp')
            if current_price and current_volume and current_time:
                if current_time not in timestamps:
                    prices.append(current_price)
                    volumes.append(current_volume)
                    timestamps.append(current_time)

            if len(prices) < self.config.min_data_points:
                self.logger.warning(
                    f"Insufficient data points: {len(prices)} < {self.config.min_data_points}"
                )
                return {}

            # 가장 최신의 min_data_points 개의 데이터만 사용
            return {
                'prices': prices[-self.config.min_data_points:],
                'volumes': volumes[-self.config.min_data_points:],
                'timestamps': timestamps[-self.config.min_data_points:]
            }
        except Exception as e:
            self.logger.error(f"Historical data collection error: {str(e)}")
            return {}

    def _calculate_risk_metrics(self, historical_data: Dict) -> Dict:
        """리스크 메트릭스 계산"""
        try:
            prices = historical_data.get('prices', [])
            volumes = historical_data.get('volumes', [])

            if len(prices) < 2:
                return {}

            # 수익률 계산
            returns = np.diff(np.log(prices))
            
            # 변동성 계산 (연율화)
            volatility = float(np.std(returns) * np.sqrt(252))
            
            # VaR 계산 (95% 신뢰수준)
            var_95 = float(np.percentile(returns, 5))
            
            # 추세 강도 계산
            ma_short = np.mean(prices[-5:])
            ma_long = np.mean(prices[-20:])
            trend_strength = float((ma_short - ma_long) / ma_long) if ma_long != 0 else 0.0

            # 거래량 프로파일
            volume_profile = self._calculate_volume_profile(volumes)

            return {
                'volatility': volatility,
                'var_95': var_95,
                'trend_strength': trend_strength,
                'volume_profile': volume_profile,
                'risk_score': self._calculate_risk_score(volatility, var_95, trend_strength)
            }

        except Exception as e:
            self.logger.error(f"Risk metrics calculation error: {str(e)}")
            return {}

    def _calculate_volume_profile(self, volumes: List[float]) -> float:
        """거래량 프로파일 계산"""
        try:
            if not volumes:
                return 0.0

            # 최근 거래량과 전체 평균 비교
            recent_volume = np.mean(volumes[-5:])
            avg_volume = np.mean(volumes)
            
            return float(recent_volume / avg_volume) if avg_volume > 0 else 1.0

        except Exception as e:
            self.logger.error(f"Volume profile calculation error: {str(e)}")
            return 1.0

    def _calculate_risk_score(self, volatility: float, var_95: float, trend_strength: float) -> float:
        """리스크 점수 계산"""
        try:
            # 변동성 기반 점수 (0~1)
            vol_score = min(1.0, volatility / self.config.max_volatility)
            
            # VaR 기반 점수 (0~1)
            var_score = min(1.0, abs(var_95) / 0.02)  # 2% 기준
            
            # 추세 강도 기반 점수 (0~1)
            trend_score = min(1.0, abs(trend_strength) / self.config.trend_threshold)
            
            # 가중 평균
            weights = [0.5, 0.3, 0.2]  # 변동성, VaR, 추세 가중치
            scores = [vol_score, var_score, trend_score]
            
            risk_score = sum(w * s for w, s in zip(weights, scores))
            return float(min(max(risk_score, 0.0), 1.0))
        except Exception as e:
            self.logger.error(f"Risk score calculation error: {str(e)}")
            return 0.5

    async def _create_base_scenario(self, context: ScenarioContext) -> Optional[MarketScenario]:
        """기본 시나리오 생성"""
        try:
            # 시장 상태 분석
            market_conditions = context.market_conditions
            volatility = market_conditions.get('volatility', 0.0)
            trend = market_conditions.get('trend', 'neutral')
            liquidity = market_conditions.get('liquidity_score', 0.0)

            # 시나리오 타입 결정
            scenario_type = self._determine_scenario_type(volatility, trend, liquidity)

            # 위험 점수 계산
            risk_score = context.risk_metrics.get('risk_score', 0.5)

            # 예상 수익률 계산
            expected_return = await self._estimate_expected_return(scenario_type, context)

            # 포지션 제안
            suggested_position = self._suggest_position(scenario_type, risk_score, expected_return)

            # 신뢰도 계산
            confidence_score = self._calculate_confidence_score(context, scenario_type)

            return MarketScenario(
                timestamp=context.timestamp,
                scenario_type=scenario_type,
                probability=self._calculate_probability(context, scenario_type),
                risk_score=risk_score,
                expected_return=expected_return,
                suggested_position=suggested_position,
                confidence_score=confidence_score,
                parameters=self._create_scenario_parameters(context)
            )
        except Exception as e:
            self.logger.error(f"Base scenario creation error: {str(e)}")
            return None

    def _determine_scenario_type(self, volatility: float, trend: str, liquidity: float) -> str:
        """시나리오 타입 결정"""
        try:
            # 변동성 기반 시나리오
            if volatility > self.config.max_volatility:
                return 'high_volatility'
            elif volatility < self.config.min_volatility:
                return 'low_volatility'
                
            # 추세 기반 시나리오
            if trend in ['strong_uptrend', 'uptrend']:
                if volatility > self.config.trend_threshold:
                    return 'strong_trend'
                return 'weak_trend'
                
            # 유동성 기반 시나리오
            if liquidity > self.config.liquidity_threshold:
                return 'high_liquidity'
            elif liquidity < self.config.liquidity_threshold * 0.5:
                return 'low_liquidity'

            return 'normal_market'
        except Exception as e:
            self.logger.error(f"Scenario type determination error: {str(e)}")
            return 'normal_market'

    async def _estimate_expected_return(self, scenario_type: str, context: ScenarioContext) -> float:
        """예상 수익률 추정"""
        try:
            base_return = self._calculate_base_return(context)
            
            # 시나리오별 조정 계수
            adjustment_factors = {
                'high_volatility': 1.5,
                'low_volatility': 0.7,
                'strong_trend': 1.3,
                'weak_trend': 0.9,
                'high_liquidity': 1.1,
                'low_liquidity': 0.8,
                'normal_market': 1.0
            }
            
            # 수익률 조정
            adjusted_return = base_return * adjustment_factors.get(scenario_type, 1.0)
            
            # 리스크 조정
            risk_adjustment = 1.0 - context.risk_metrics.get('risk_score', 0.5)
            
            return float(adjusted_return * risk_adjustment)
        except Exception as e:
            self.logger.error(f"Expected return estimation error: {str(e)}")
            return 0.0

    def _suggest_position(self, scenario_type: str, risk_score: float, expected_return: float) -> str:
        """포지션 제안"""
        try:
            # 리스크가 너무 높으면 중립
            if risk_score > 0.8:
                return 'neutral'

            # 예상 수익률이 너무 낮으면 중립
            if abs(expected_return) < 0.001:  # 0.1% 미만
                return 'neutral'

            # 시나리오별 포지션 결정
            if scenario_type in ['strong_trend', 'high_liquidity']:
                return 'long' if expected_return > 0 else 'short'
            elif scenario_type in ['high_volatility']:
                return 'neutral'  # 변동성이 높을 때는 중립
            else:
                # 일반적인 상황에서는 예상 수익률에 따라
                if expected_return > 0.002:  # 0.2% 이상
                    return 'long'
                elif expected_return < -0.002:
                    return 'short'
                return 'neutral'
        except Exception as e:
            self.logger.error(f"Position suggestion error: {str(e)}")
            return 'neutral'

    def _calculate_confidence_score(self, context: ScenarioContext, scenario_type: str) -> float:
        """신뢰도 계산"""
        try:
            # 신뢰도는 위험 점수와 예상 수익률, 과거 성과 등을 기반으로 계산
            # 여기서는 간단한 예시로 0.5 고정값을 반환
            return 0.5
        except Exception as e:
            self.logger.error(f"Confidence score calculation error: {str(e)}")
            return 0.5

    def _calculate_probability(self, context: ScenarioContext, scenario_type: str) -> float:
        """시나리오 확률 계산"""
        try:
            # 기본 확률 계산 (예시)
            probability = random.uniform(0, 1)
            return probability
        except Exception as e:
            self.logger.error(f"Probability calculation error: {str(e)}")
            return 0.0

    def _create_scenario_parameters(self, context: ScenarioContext) -> Dict:
        """시나리오 파라미터 생성"""
        try:
            return {
                'market_conditions': {
                    'trend': context.market_conditions.get('trend', 'neutral'),
                    'volatility': context.market_conditions.get('volatility', 0.0),
                    'liquidity': context.market_conditions.get('liquidity_score', 0.0)
                },
                'risk_metrics': context.risk_metrics,
                'performance_stats': context.performance_stats or {},
                'timestamp': context.timestamp.isoformat()
            }
        except Exception as e:
            self.logger.error(f"Parameter creation error: {str(e)}")
            return {}

    def _calculate_base_return(self, context: ScenarioContext) -> float:
        """기본 수익률 계산 (예시 구현)"""
        try:
            # 간단한 예시로 과거 가격 변화율의 평균을 수익률로 사용
            prices = context.historical_data.get('prices', [])
            if len(prices) < 2:
                return 0.0
            returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
            return float(sum(returns) / len(returns))
        except Exception as e:
            self.logger.error(f"Base return calculation error: {str(e)}")
            return 0.0
