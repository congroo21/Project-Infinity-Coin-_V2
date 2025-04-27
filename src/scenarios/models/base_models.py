# src/scenarios/models/base_models.py

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

@dataclass
class MarketScenario:
    """시나리오 데이터 클래스"""
    timestamp: datetime
    scenario_type: str  # 'volatility', 'trend', 'liquidity'
    probability: float  # 발생 확률
    risk_score: float  # 리스크 점수 (0~1)
    expected_return: float  # 예상 수익률
    suggested_position: str  # 'long', 'short', 'neutral'
    confidence_score: float  # 신뢰도 점수 (0~1)
    parameters: Dict  # 시나리오별 특수 파라미터
    
    # 확률 분석 결과
    bayesian_probability: Optional[float] = None
    monte_carlo_var95: Optional[float] = None
    monte_carlo_expected_return: Optional[float] = None
    monte_carlo_sharpe: Optional[float] = None
    monte_carlo_confidence: Optional[float] = None

@dataclass
class AnalysisResult:
    """통합 분석 결과 데이터 클래스"""
    scenario: MarketScenario
    probability_metrics: Dict
    risk_metrics: Dict
    simulation_metrics: Dict
    timestamp: datetime
    
    def validate(self) -> bool:
        """분석 결과 유효성 검증"""
        try:
            required_metrics = {
                'probability_metrics': ['bayesian_prob', 'confidence'],
                'risk_metrics': ['var95', 'max_drawdown'],
                'simulation_metrics': ['expected_return', 'sharpe_ratio']
            }
            
            for category, metrics in required_metrics.items():
                data = getattr(self, category)
                if not all(metric in data for metric in metrics):
                    return False
            return True
        except Exception:
            return False