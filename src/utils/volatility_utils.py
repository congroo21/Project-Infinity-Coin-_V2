# src/utils/volatility_utils.py

import numpy as np
import logging
from typing import List, Dict, Optional, Union, Tuple

# risk_utils에서 리스크 계산기 가져오기
from src.utils.risk_utils import risk_calculator

logger = logging.getLogger(__name__)

# 기존 함수는 유지하되 risk_calculator를 활용하도록 수정

def calculate_returns(prices: List[float]) -> np.ndarray:
    """
    가격 데이터에서 로그 수익률 계산
    
    Args:
        prices: 가격 데이터 리스트
        
    Returns:
        로그 수익률 배열
    """
    return risk_calculator.calculate_returns(prices)

def calculate_volatility(
    prices: List[float], 
    annualize: bool = True, 
    trading_days: int = 252
) -> float:
    """
    가격 데이터에서 변동성 계산
    
    Args:
        prices: 가격 데이터 리스트
        annualize: 연간화 여부 (기본값: True)
        trading_days: 연간 거래일 수 (기본값: 252)
        
    Returns:
        변동성 값 (0.0 ~ 1.0 사이의 값)
    """
    try:
        if len(prices) < 2:
            return 0.0
            
        # 수익률 계산
        returns = calculate_returns(prices)
        
        # 중앙 집중화된 리스크 계산기 사용
        old_factor = risk_calculator.annualization_factor
        risk_calculator.annualization_factor = trading_days
        volatility = risk_calculator.calculate_volatility(returns, annualize)
        risk_calculator.annualization_factor = old_factor
            
        return float(volatility)
    except Exception as e:
        logger.error(f"변동성 계산 오류: {e}")
        return 0.0

def calculate_returns(prices: List[float]) -> np.ndarray:
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
        
        # 로그 수익률 계산
        return np.diff(np.log(prices))
    except Exception as e:
        logger.error(f"수익률 계산 오류: {e}")
        return np.array([])

def calculate_volatility(
    prices: List[float], 
    annualize: bool = True, 
    trading_days: int = 252
) -> float:
    """
    가격 데이터에서 변동성 계산
    
    Args:
        prices: 가격 데이터 리스트
        annualize: 연간화 여부 (기본값: True)
        trading_days: 연간 거래일 수 (기본값: 252)
        
    Returns:
        변동성 값 (0.0 ~ 1.0 사이의 값)
    """
    try:
        if len(prices) < 2:
            return 0.0
            
        # 수익률 계산
        returns = calculate_returns(prices)
        
        # 표준편차 계산
        volatility = np.std(returns)
        
        # 연간화 (필요시)
        if annualize and volatility > 0:
            volatility *= np.sqrt(trading_days)
            
        return float(volatility)
    except Exception as e:
        logger.error(f"변동성 계산 오류: {e}")
        return 0.0

def calculate_garch_volatility(
    returns: np.ndarray, 
    omega: float = 0.00001, 
    alpha: float = 0.1, 
    beta: float = 0.8,
    last_vol: float = None
) -> float:
    """
    GARCH(1,1) 모델을 사용한 변동성 계산
    
    Args:
        returns: 수익률 배열
        omega: 기본 변동성 파라미터
        alpha: 최근 충격 가중치
        beta: 과거 변동성 가중치
        last_vol: 이전 변동성 값 (None인 경우 returns의 표준편차 사용)
        
    Returns:
        GARCH 기반 변동성 값
    """
    try:
        if len(returns) == 0:
            return 0.0
            
        # 마지막 수익률
        last_return = returns[-1]
        
        # 이전 변동성이 없으면 표준편차 사용
        if last_vol is None:
            last_vol = np.std(returns)
            
        # GARCH(1,1) 업데이트
        new_vol = np.sqrt(
            omega +
            alpha * (last_return ** 2) +
            beta * (last_vol ** 2)
        )
        
        return float(new_vol)
    except Exception as e:
        logger.error(f"GARCH 변동성 계산 오류: {e}")
        return 0.0

def calculate_ma_volatility(
    prices: List[float], 
    window: int = 20, 
    alpha: float = 0.1
) -> float:
    """
    이동평균 기반 변동성 계산
    
    Args:
        prices: 가격 데이터 리스트
        window: 이동평균 기간
        alpha: EMA 가중치
        
    Returns:
        이동평균 기반 변동성 값
    """
    try:
        if len(prices) < window:
            return 0.0
            
        # 수익률 계산
        returns = calculate_returns(prices)
        
        # 지수이동평균(EMA) 계산
        if len(returns) == 0:
            return 0.0
            
        ema = returns[0]
        for ret in returns[1:]:
            ema = alpha * ret + (1 - alpha) * ema
            
        return float(ema)
    except Exception as e:
        logger.error(f"MA 변동성 계산 오류: {e}")
        return 0.0
    # src/utils/volatility_utils.py에 추가

class VolatilityCalculator:
    """변동성 계산을 위한 클래스"""
    
    def __init__(self, method: str = 'standard', **kwargs):
        """
        변동성 계산기 초기화
        
        Args:
            method: 계산 방법 ('standard', 'garch', 'ma' 중 하나)
            **kwargs: 추가 파라미터
        """
        self.method = method
        self.params = kwargs
        self.history = []
        
        # GARCH 모델 파라미터
        self.omega = kwargs.get('omega', 0.00001)
        self.alpha = kwargs.get('alpha', 0.1)
        self.beta = kwargs.get('beta', 0.8)
        
        # 이동평균 파라미터
        self.ma_window = kwargs.get('ma_window', 20)
        self.alpha_ma = kwargs.get('alpha_ma', 0.1)
        
        # 연간화 설정
        self.annualize = kwargs.get('annualize', True)
        self.trading_days = kwargs.get('trading_days', 252)
        
        # 마지막 계산 결과
        self.last_volatility = 0.0
    
    def update(self, price: float) -> float:
        """
        새 가격으로 변동성 업데이트
        
        Args:
            price: 새 가격
            
        Returns:
            업데이트된 변동성 값
        """
        self.history.append(price)
        
        # 계산에 필요한 최소 데이터 확인
        if len(self.history) < 2:
            return 0.0
        
        # 선택한 방법으로 변동성 계산
        if self.method == 'garch':
            returns = calculate_returns(self.history)
            if len(returns) > 0:
                self.last_volatility = calculate_garch_volatility(
                    returns, 
                    self.omega, 
                    self.alpha, 
                    self.beta,
                    self.last_volatility
                )
        elif self.method == 'ma':
            self.last_volatility = calculate_ma_volatility(
                self.history,
                self.ma_window,
                self.alpha_ma
            )
        else:  # 'standard'
            self.last_volatility = calculate_volatility(
                self.history,
                self.annualize,
                self.trading_days
            )
            
        return self.last_volatility
    
    def get_current_volatility(self) -> float:
        """현재 변동성 값 반환"""
        return self.last_volatility
    
    def reset(self):
        """히스토리 및 계산 초기화"""
        self.history = []
        self.last_volatility = 0.0