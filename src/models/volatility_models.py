# src/models/volatility_models.py

import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from ..utils.volatility_utils import calculate_returns, calculate_garch_volatility, calculate_ma_volatility

@dataclass
class VolatilityMetrics:
    """변동성 메트릭스"""
    timestamp: datetime
    realized_vol: float      # 실현 변동성
    implied_vol: float       # 내재 변동성
    predicted_vol: float     # 예측 변동성
    regime: str             # 변동성 국면
    confidence: float       # 예측 신뢰도

class EnhancedVolatilityModel:
    """개선된 변동성 모델"""
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.price_history = deque(maxlen=lookback_window)
        self.vol_history = deque(maxlen=lookback_window)
        
        # GARCH 모델 파라미터
        self.omega = 0.00001    # 기본 변동성
        self.alpha = 0.1        # 최근 충격 가중치
        self.beta = 0.8         # 과거 변동성 가중치
        
        # 변동성 국면 임계값
        self.regime_thresholds = {
            'very_low': 0.005,   # 0.5%
            'low': 0.01,         # 1%
            'medium': 0.02,      # 2%
            'high': 0.03,        # 3%
            'very_high': 0.05    # 5%
        }
        
        # 캐시 및 상태 관리
        self.last_update = None
        self.current_regime = 'medium'
        self.model_parameters = {}
        
    def update(self, price: float, timestamp: datetime = None):
        """가격 데이터 업데이트"""
        try:
            if not timestamp:
                timestamp = datetime.now()
            
            self.price_history.append((timestamp, price))
            
            # 수익률 계산 및 변동성 업데이트
            if len(self.price_history) >= 2:
                returns = self._calculate_returns()
                current_vol = self._update_garch(returns[-1])
                self.vol_history.append((timestamp, current_vol))
                
                # 변동성 국면 업데이트
                self.current_regime = self._determine_regime(current_vol)
                
            self.last_update = timestamp
            
        except Exception as e:
            logging.error(f"변동성 업데이트 오류: {e}")

    def get_current_metrics(self) -> Optional[VolatilityMetrics]:
        """현재 변동성 메트릭스 조회"""
        try:
            if not self.vol_history:
                return None
                
            current_vol = self.vol_history[-1][1]
            predicted_vol = self._predict_volatility()
            
            return VolatilityMetrics(
                timestamp=datetime.now(),
                realized_vol=current_vol,
                implied_vol=self._estimate_implied_vol(),
                predicted_vol=predicted_vol,
                regime=self.current_regime,
                confidence=self._calculate_prediction_confidence()
            )
            
        except Exception as e:
            logging.error(f"변동성 메트릭스 계산 오류: {e}")
            return None

    def _calculate_returns(self) -> np.ndarray:
        """수익률 계산"""
        prices = [p[1] for p in self.price_history]
        return calculate_returns(prices)

    def _update_garch(self, return_value: float) -> float:
        """GARCH 모델 업데이트"""
        try:
            if not self.vol_history:
                return abs(return_value)
                
            last_vol = self.vol_history[-1][1]
            
            # 유틸리티 함수 사용
            new_vol = calculate_garch_volatility(
                np.array([return_value]), 
                self.omega, 
                self.alpha, 
                self.beta,
                last_vol
            )
            
            # 모델 파라미터 동적 조정
            self._adjust_parameters(return_value, new_vol)
            
            return new_vol
            
        except Exception as e:
            logging.error(f"GARCH 업데이트 오류: {e}")
            return abs(return_value)


    def _adjust_parameters(self, return_value: float, current_vol: float):
        """모델 파라미터 동적 조정"""
        try:
            # 예측 오차 계산
            if len(self.vol_history) >= 2:
                last_predicted = self.vol_history[-2][1]
                prediction_error = abs(current_vol - last_predicted)
                
                # 오차가 큰 경우 파라미터 조정
                if prediction_error > 0.01:  # 1% 이상 오차
                    # alpha 증가 (최근 데이터 가중치 증가)
                    self.alpha = min(0.2, self.alpha * 1.1)
                    # beta 감소 (과거 데이터 가중치 감소)
                    self.beta = max(0.7, self.beta * 0.95)
                else:
                    # 점진적으로 기본값으로 복귀
                    self.alpha = 0.1 + (self.alpha - 0.1) * 0.95
                    self.beta = 0.8 + (self.beta - 0.8) * 0.95
                    
        except Exception as e:
            logging.error(f"파라미터 조정 오류: {e}")

    def _predict_volatility(self) -> float:
        """변동성 예측"""
        try:
            if not self.vol_history:
                return 0.0
                
            # 기본 GARCH 예측
            current_vol = self.vol_history[-1][1]
            garch_prediction = np.sqrt(
                self.omega +
                self.alpha * (current_vol ** 2) +
                self.beta * (current_vol ** 2)
            )
            
            # 이동평균 기반 예측
            ma_prediction = self._calculate_ma_volatility()
            
            # 가중 평균으로 최종 예측
            weights = self._calculate_model_weights()
            final_prediction = (
                weights['garch'] * garch_prediction +
                weights['ma'] * ma_prediction
            )
            
            return final_prediction
            
        except Exception as e:
            logging.error(f"변동성 예측 오류: {e}")
            return self.vol_history[-1][1] if self.vol_history else 0.0

    def _calculate_ma_volatility(self) -> float:
        """이동평균 기반 변동성 계산"""
        try:
            if len(self.vol_history) < 2:
                return 0.0
                
            recent_vols = [v[1] for v in self.vol_history]
            
            # 유틸리티 함수 사용 (여기서는 가격 대신 변동성 값을 사용)
            # 이 경우 calculate_ma_volatility는 내부적으로 수익률을 계산하지 않음
            alpha = 0.1
            ema = recent_vols[0]
            for vol in recent_vols[1:]:
                ema = alpha * vol + (1 - alpha) * ema
                
            return ema
            
        except Exception as e:
            logging.error(f"MA 변동성 계산 오류: {e}")
            return 0.0

    def _calculate_model_weights(self) -> Dict[str, float]:
        """모델별 가중치 계산"""
        try:
            # 기본 가중치
            weights = {'garch': 0.7, 'ma': 0.3}
            
            if len(self.vol_history) >= 30:
                # 최근 예측 정확도로 가중치 조정
                garch_accuracy = self._calculate_model_accuracy('garch')
                ma_accuracy = self._calculate_model_accuracy('ma')
                
                total_accuracy = garch_accuracy + ma_accuracy
                if total_accuracy > 0:
                    weights['garch'] = garch_accuracy / total_accuracy
                    weights['ma'] = ma_accuracy / total_accuracy
                    
            return weights
            
        except Exception as e:
            logging.error(f"모델 가중치 계산 오류: {e}")
            return {'garch': 0.7, 'ma': 0.3}

    def _calculate_model_accuracy(self, model_type: str) -> float:
        """모델별 예측 정확도 계산"""
        try:
            if len(self.vol_history) < 30:
                return 1.0
                
            recent_vols = [v[1] for v in self.vol_history[-30:]]
            predictions = []
            
            # 각 모델의 과거 예측값 계산
            if model_type == 'garch':
                for i in range(len(recent_vols)-1):
                    pred = np.sqrt(
                        self.omega +
                        self.alpha * (recent_vols[i] ** 2) +
                        self.beta * (recent_vols[i] ** 2)
                    )
                    predictions.append(pred)
            else:  # ma
                ema = recent_vols[0]
                for vol in recent_vols[1:]:
                    pred = ema
                    ema = 0.1 * vol + 0.9 * ema
                    predictions.append(pred)
            
            # RMSE 계산 및 정확도 변환
            errors = np.array(predictions) - np.array(recent_vols[1:])
            rmse = np.sqrt(np.mean(errors ** 2))
            
            # RMSE를 0~1 범위의 정확도로 변환
            accuracy = 1 / (1 + rmse)
            return accuracy
            
        except Exception as e:
            logging.error(f"모델 정확도 계산 오류: {e}")
            return 1.0

    def _determine_regime(self, volatility: float) -> str:
        """변동성 국면 판단"""
        try:
            if volatility <= self.regime_thresholds['very_low']:
                return 'very_low'
            elif volatility <= self.regime_thresholds['low']:
                return 'low'
            elif volatility <= self.regime_thresholds['medium']:
                return 'medium'
            elif volatility <= self.regime_thresholds['high']:
                return 'high'
            else:
                return 'very_high'
                
        except Exception as e:
            logging.error(f"변동성 국면 판단 오류: {e}")
            return 'medium'

    def _estimate_implied_vol(self) -> float:
        """내재 변동성 추정"""
        try:
            if not self.vol_history:
                return 0.0
                
            # 실제 구현 시에는 옵션 데이터로부터 계산
            # 여기서는 실현 변동성의 1.1배로 가정
            return self.vol_history[-1][1] * 1.1
            
        except Exception as e:
            logging.error(f"내재 변동성 추정 오류: {e}")
            return 0.0

    def _calculate_prediction_confidence(self) -> float:
        """예측 신뢰도 계산"""
        try:
            if len(self.vol_history) < 30:
                return 0.5
                
            # 최근 예측 정확도로 신뢰도 계산
            garch_accuracy = self._calculate_model_accuracy('garch')
            ma_accuracy = self._calculate_model_accuracy('ma')
            
            # 가중 평균 신뢰도
            weights = self._calculate_model_weights()
            confidence = (
                weights['garch'] * garch_accuracy +
                weights['ma'] * ma_accuracy
            )
            
            return confidence
            
        except Exception as e:
            logging.error(f"신뢰도 계산 오류: {e}")
            return 0.5