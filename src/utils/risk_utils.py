# src/utils/risk_utils.py

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import logging

class RiskMetricsCalculator:
    """중앙화된 리스크 메트릭스 계산 클래스"""
    
    def __init__(self, 
                 confidence_levels: List[float] = [0.95, 0.99],
                 annualization_factor: float = 252,
                 risk_free_rate: float = 0.02):
        """
        리스크 메트릭스 계산기 초기화
        
        Args:
            confidence_levels: VaR 계산에 사용할 신뢰 수준 리스트 (기본값: 95%, 99%)
            annualization_factor: 연간화 계수 (기본값: 252 거래일)
            risk_free_rate: 무위험 수익률 (기본값: 2%)
        """
        self.confidence_levels = confidence_levels
        self.annualization_factor = annualization_factor
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)

    def calculate_returns(self, prices: List[float]) -> np.ndarray:
        """
        가격 데이터에서 로그 수익률 계산
        
        Args:
            prices: 가격 데이터 리스트
            
        Returns:
            로그 수익률 배열
        """
        try:
            if len(prices) < 2:
                return np.array([])
            
            return np.diff(np.log(prices))
        except Exception as e:
            self.logger.error(f"수익률 계산 오류: {e}")
            return np.array([])

    def calculate_volatility(self, 
                            returns: Union[List[float], np.ndarray], 
                            annualize: bool = True) -> float:
        """
        수익률 데이터에서 변동성 계산
        
        Args:
            returns: 수익률 데이터
            annualize: 연간화 여부
            
        Returns:
            변동성 값
        """
        try:
            if isinstance(returns, list) and len(returns) < 2:
                return 0.0
            elif isinstance(returns, np.ndarray) and returns.size < 2:
                return 0.0
                
            # 표준편차 계산
            vol = float(np.std(returns))
            
            # 연간화 (필요시)
            if annualize and vol > 0:
                vol *= np.sqrt(self.annualization_factor)
                
            return vol
        except Exception as e:
            self.logger.error(f"변동성 계산 오류: {e}")
            return 0.0

    def calculate_parametric_var(self, 
                               returns: Union[List[float], np.ndarray], 
                               confidence_level: float = 0.95) -> float:
        """
        파라메트릭 VaR 계산
        
        Args:
            returns: 수익률 데이터
            confidence_level: 신뢰 수준 (기본값: 95%)
            
        Returns:
            VaR 값
        """
        try:
            if isinstance(returns, list) and len(returns) < 2:
                return 0.0
            elif isinstance(returns, np.ndarray) and returns.size < 2:
                return 0.0

            mean = np.mean(returns)
            std = np.std(returns)
            
            # 정규분포 가정
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean + z_score * std)
            
            return float(var)
        except Exception as e:
            self.logger.error(f"파라메트릭 VaR 계산 오류: {e}")
            return 0.0

    def calculate_historical_var(self, 
                                returns: Union[List[float], np.ndarray], 
                                confidence_level: float = 0.95) -> float:
        """
        히스토리컬 VaR 계산
        
        Args:
            returns: 수익률 데이터
            confidence_level: 신뢰 수준 (기본값: 95%)
            
        Returns:
            VaR 값
        """
        try:
            if isinstance(returns, list) and len(returns) < 2:
                return 0.0
            elif isinstance(returns, np.ndarray) and returns.size < 2:
                return 0.0
                
            # 히스토리컬 시뮬레이션 (비파라메트릭 접근법)
            sorted_returns = np.sort(returns)
            index = int(len(sorted_returns) * (1 - confidence_level))
            var = -sorted_returns[index]
            
            return float(var)
        except Exception as e:
            self.logger.error(f"히스토리컬 VaR 계산 오류: {e}")
            return 0.0

    def calculate_conditional_var(self, 
                                 returns: Union[List[float], np.ndarray], 
                                 confidence_level: float = 0.95) -> float:
        """
        조건부 VaR (CVaR / Expected Shortfall) 계산
        
        Args:
            returns: 수익률 데이터
            confidence_level: 신뢰 수준 (기본값: 95%)
            
        Returns:
            CVaR 값
        """
        try:
            if isinstance(returns, list) and len(returns) < 2:
                return 0.0
            elif isinstance(returns, np.ndarray) and returns.size < 2:
                return 0.0
                
            sorted_returns = np.sort(returns)
            var_index = int(len(sorted_returns) * (1 - confidence_level))
            
            # VaR 이하의 모든 값의 평균
            cvar = -np.mean(sorted_returns[:var_index+1])
            
            return float(cvar)
        except Exception as e:
            self.logger.error(f"조건부 VaR 계산 오류: {e}")
            return 0.0

    def calculate_sharpe_ratio(self, 
                              returns: Union[List[float], np.ndarray],
                              risk_free_rate: Optional[float] = None,
                              annualize: bool = True) -> float:
        """
        샤프 비율 계산
        
        Args:
            returns: 수익률 데이터
            risk_free_rate: 무위험 수익률 (None이면 초기화 시 설정한 값 사용)
            annualize: 연간화 여부
            
        Returns:
            샤프 비율
        """
        try:
            if isinstance(returns, list) and len(returns) < 2:
                return 0.0
            elif isinstance(returns, np.ndarray) and returns.size < 2:
                return 0.0
                
            # 무위험 수익률 결정
            rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            
            # 일일 무위험 수익률로 변환
            daily_rf = rf / self.annualization_factor
            
            # 초과 수익률 계산
            excess_returns = np.array(returns) - daily_rf
            
            # 샤프 비율 계산
            sharpe = np.mean(excess_returns) / np.std(excess_returns)
            
            # 연간화 (필요시)
            if annualize:
                sharpe *= np.sqrt(self.annualization_factor)
                
            return float(sharpe)
        except Exception as e:
            self.logger.error(f"샤프 비율 계산 오류: {e}")
            return 0.0

    def calculate_max_drawdown(self, returns: Union[List[float], np.ndarray]) -> float:
        """
        최대 낙폭 계산
        
        Args:
            returns: 수익률 데이터
            
        Returns:
            최대 낙폭 값
        """
        try:
            if isinstance(returns, list) and len(returns) < 2:
                return 0.0
            elif isinstance(returns, np.ndarray) and returns.size < 2:
                return 0.0
                
            # 누적 수익률 계산
            cumulative_returns = np.cumprod(1 + np.array(returns))
            
            # 현재까지의 최대값 계산
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # 낙폭 계산
            drawdowns = (running_max - cumulative_returns) / running_max
            
            # 최대 낙폭
            max_drawdown = np.max(drawdowns)
            
            return float(max_drawdown)
        except Exception as e:
            self.logger.error(f"최대 낙폭 계산 오류: {e}")
            return 0.0

    def calculate_correlation(self, 
                             returns_a: Union[List[float], np.ndarray], 
                             returns_b: Union[List[float], np.ndarray]) -> float:
        """
        두 자산 간 상관관계 계산
        
        Args:
            returns_a: 첫 번째 자산의 수익률 데이터
            returns_b: 두 번째 자산의 수익률 데이터
            
        Returns:
            상관계수 (-1 ~ 1)
        """
        try:
            if (isinstance(returns_a, list) and len(returns_a) < 2) or \
               (isinstance(returns_b, list) and len(returns_b) < 2):
                return 0.0
            elif (isinstance(returns_a, np.ndarray) and returns_a.size < 2) or \
                 (isinstance(returns_b, np.ndarray) and returns_b.size < 2):
                return 0.0
                
            # 상관계수 계산
            corr = np.corrcoef(returns_a, returns_b)[0, 1]
            
            return float(corr)
        except Exception as e:
            self.logger.error(f"상관관계 계산 오류: {e}")
            return 0.0

    def calculate_beta(self, 
                      returns_asset: Union[List[float], np.ndarray], 
                      returns_market: Union[List[float], np.ndarray]) -> float:
        """
        베타 계산
        
        Args:
            returns_asset: 자산의 수익률 데이터
            returns_market: 시장의 수익률 데이터
            
        Returns:
            베타 값
        """
        try:
            if (isinstance(returns_asset, list) and len(returns_asset) < 2) or \
               (isinstance(returns_market, list) and len(returns_market) < 2):
                return 1.0
            elif (isinstance(returns_asset, np.ndarray) and returns_asset.size < 2) or \
                 (isinstance(returns_market, np.ndarray) and returns_market.size < 2):
                return 1.0
                
            # 공분산 계산
            cov = np.cov(returns_asset, returns_market)[0, 1]
            
            # 시장 분산 계산
            market_var = np.var(returns_market)
            
            # 베타 계산
            beta = cov / market_var if market_var > 0 else 1.0
            
            return float(beta)
        except Exception as e:
            self.logger.error(f"베타 계산 오류: {e}")
            return 1.0

    def calculate_herfindahl_index(self, weights: List[float]) -> float:
        """
        허핀달 지수 계산 (포트폴리오 집중도)
        
        Args:
            weights: 포트폴리오 자산 비중 리스트
            
        Returns:
            허핀달 지수 (0 ~ 1)
        """
        try:
            if not weights:
                return 0.0
                
            # 허핀달 지수 계산
            h_index = sum(w * w for w in weights)
            
            return float(h_index)
        except Exception as e:
            self.logger.error(f"허핀달 지수 계산 오류: {e}")
            return 0.0

    def calculate_var_levels(self, returns: Union[List[float], np.ndarray]) -> Dict[str, float]:
        """
        다양한 VaR 수준 계산
        
        Args:
            returns: 수익률 데이터
            
        Returns:
            다양한 VaR 값을 포함하는 딕셔너리
        """
        try:
            result = {}
            
            for cl in self.confidence_levels:
                cl_str = str(int(cl * 100))
                
                # 파라메트릭 VaR
                result[f'parametric_var_{cl_str}'] = self.calculate_parametric_var(returns, cl)
                
                # 히스토리컬 VaR
                result[f'historical_var_{cl_str}'] = self.calculate_historical_var(returns, cl)
                
                # 조건부 VaR
                result[f'conditional_var_{cl_str}'] = self.calculate_conditional_var(returns, cl)
            
            return result
        except Exception as e:
            self.logger.error(f"VaR 수준 계산 오류: {e}")
            return {
                'parametric_var_95': 0.0,
                'historical_var_95': 0.0,
                'conditional_var_95': 0.0,
                'parametric_var_99': 0.0,
                'historical_var_99': 0.0,
                'conditional_var_99': 0.0
            }

    def calculate_risk_metrics(self, returns: Union[List[float], np.ndarray]) -> Dict[str, float]:
        """
        종합 리스크 메트릭스 계산
        
        Args:
            returns: 수익률 데이터
            
        Returns:
            다양한 리스크 메트릭스를 포함하는 딕셔너리
        """
        try:
            metrics = {
                'volatility': self.calculate_volatility(returns),
                'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                'max_drawdown': self.calculate_max_drawdown(returns)
            }
            
            # VaR 수준 추가
            metrics.update(self.calculate_var_levels(returns))
            
            # 통계적 지표 추가
            if len(returns) >= 2:
                metrics['kurtosis'] = float(stats.kurtosis(returns))
                metrics['skewness'] = float(stats.skew(returns))
            
            return metrics
        except Exception as e:
            self.logger.error(f"리스크 메트릭스 계산 오류: {e}")
            return {
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'parametric_var_95': 0.0,
                'historical_var_95': 0.0
            }

# 싱글톤 인스턴스
risk_calculator = RiskMetricsCalculator()