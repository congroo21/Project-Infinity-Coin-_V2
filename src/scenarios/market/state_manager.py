#src/scenarios/market/state_manager.py

import logging
import numpy as np
from collections import deque
from typing import Dict, Optional
from datetime import datetime
import asyncio

from src.models.market_state import MarketState, MarketMetrics
from ...exceptions import ValidationError, InsufficientDataError
from ...config import Config

class MarketStateManager:
    """개선된 시장 상태 관리 클래스"""
    def __init__(self, window_size: int = 1000):
        # 데이터 저장소
        self.price_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
        self.volatility_history = deque(maxlen=window_size)
        self.liquidity_history = deque(maxlen=window_size)
        self.integrated_history = deque(maxlen=window_size)

        # 시간대별 상태 저장소
        self.timeframe_states = {
            tf: {'history': deque(maxlen=window_size)} 
            for tf in Config.TIMEFRAMES
        }
        
        # 임계값 설정
        self.thresholds = Config.RISK_THRESHOLDS
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)

    def update(self, market_data: Dict) -> None:
        """시장 상태 업데이트"""
        try:
            # 데이터 타입 검증 추가
            if not isinstance(market_data.get('timestamp'), datetime):
                raise ValidationError("Invalid timestamp type")
            if not isinstance(market_data.get('price'), (int, float)) or market_data.get('price', 0) <= 0:
                raise ValidationError("Invalid price value")
            if not isinstance(market_data.get('volume'), (int, float)) or market_data.get('volume', 0) < 0:
                raise ValidationError("Invalid volume value")
            if market_data.get('timeframe') not in ['1m', '5m', '15m', '1h', '4h', '1d']:
                raise ValidationError("Invalid timeframe")
            
            # 데이터 유효성 검증
            if not self._validate_market_data(market_data):
                self.logger.warning("Invalid market data received")
                return

            # 시장 상태 객체 생성
            state = MarketState(
                timestamp=datetime.now(),
                price=market_data['price'],
                volume=market_data['volume'],
                timeframe=market_data.get('timeframe', '1m'),
                orderbook=market_data.get('orderbook')
            )

            # 기본 데이터 업데이트
            self._update_basic_data(state)
            
            # 시장 지표 계산
            state.calculate_metrics(
                list(self.price_history),
                list(self.volume_history)
            )
            
            # 추세 판단
            self._determine_trend(state)
            
            # 상태 저장
            self._save_state(state)
            
            # 급격한 변화 감지
            if self._detect_sudden_change(state):
                self._trigger_alert(state)

        except Exception as e:
            self.logger.error(f"Market state update error: {e}")
            raise

    def _validate_market_data(self, data: Dict) -> bool:
        """시장 데이터 유효성 검증"""
        required_fields = ['price', 'volume']
        if not all(field in data for field in required_fields):
            return False
            
        try:
            price = float(data['price'])
            volume = float(data['volume'])
            return price > 0 and volume >= 0
        except (ValueError, TypeError):
            return False

    def _update_basic_data(self, state: MarketState) -> None:
        """기본 데이터 업데이트"""
        try:
            self.price_history.append(state.price)
            self.volume_history.append(state.volume)
            
            # 유동성 점수 계산 (orderbook이 있는 경우)
            if state.orderbook:
                liquidity = self._calculate_liquidity(state.orderbook)
                self.liquidity_history.append(liquidity)
                
        except Exception as e:
            self.logger.error(f"Basic data update error: {e}")
            raise

    def _determine_trend(self, state: MarketState) -> None:
        """추세 판단"""
        try:
            if len(self.price_history) < 20:
                state.trend = 'neutral'
                return

            prices = list(self.price_history)
            ma_short = np.mean(prices[-5:])
            ma_long = np.mean(prices[-20:])
            
            if ma_short > ma_long * (1 + self.thresholds['trend_strong']):
                state.trend = 'strong_uptrend'
            elif ma_short > ma_long:
                state.trend = 'uptrend'
            elif ma_short < ma_long * (1 - self.thresholds['trend_strong']):
                state.trend = 'strong_downtrend'
            elif ma_short < ma_long:
                state.trend = 'downtrend'
            else:
                state.trend = 'neutral'

        except Exception as e:
            self.logger.error(f"Trend determination error: {e}")
            state.trend = 'neutral'

    def _save_state(self, state: MarketState) -> None:
        """상태 저장"""
        try:
            # 시간대별 상태 저장
            if state.timeframe in self.timeframe_states:
                self.timeframe_states[state.timeframe]['history'].append(state)
            
            # 통합 상태 저장
            self.integrated_history.append(state)
            
        except Exception as e:
            self.logger.error(f"State saving error: {e}")
            raise

    def _detect_sudden_change(self, state: MarketState) -> bool:
        """급격한 변화 감지"""
        if len(self.integrated_history) < 2:
            return False
            
        prev_state = self.integrated_history[-2]
        
        # 변동성 급증
        if (state.metrics.volatility > 
            prev_state.metrics.volatility * 1.5):
            return True
            
        # 가격 급등락
        if abs(state.price - prev_state.price) / prev_state.price > 0.05:
            return True
            
        return False

    def _trigger_alert(self, state: MarketState) -> None:
        """급격한 변화 발생 시 알림"""
        self.logger.warning(
            f"Sudden market change detected! "
            f"Current state: {state.to_dict()}"
        )

    def _calculate_liquidity(self, orderbook: Dict) -> float:
        """유동성 점수 계산"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            total_bid_volume = sum(order.get('size', 0) for order in bids)
            total_ask_volume = sum(order.get('size', 0) for order in asks)
            
            return min(1.0, (total_bid_volume + total_ask_volume) / 
                      self.thresholds['liquidity_high'])
                      
        except Exception as e:
            self.logger.error(f"Liquidity calculation error: {e}")
            return 0.0

    def get_current_state(self) -> Dict:
        """현재 시장 상태 조회"""
        try:
            if not self.integrated_history:
                raise InsufficientDataError("No market state available")
                
            current_state = self.integrated_history[-1]
            return current_state.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error getting current state: {e}")
            raise

    def get_timeframe_state(self, timeframe: str) -> Optional[Dict]:
        """특정 시간대의 상태 조회"""
        try:
            if timeframe not in self.timeframe_states:
                raise ValidationError(f"Invalid timeframe: {timeframe}")
                
            history = self.timeframe_states[timeframe]['history']
            if not history:
                return None
                
            return history[-1].to_dict()
            
        except Exception as e:
            self.logger.error(f"Error getting timeframe state: {e}")
            return None

    async def start_monitoring(self):
        """상태 모니터링 시작"""
        while True:
            try:
                current_state = self.get_current_state()
                self.logger.info(f"Current market state: {current_state}")
                await asyncio.sleep(Config.MONITORING_INTERVAL)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)  # 에러 발생 시 5초 대기