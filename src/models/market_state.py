# src/models/market_state.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from src.exceptions import ValidationError
from ..utils.volatility_utils import calculate_volatility

@dataclass
class MarketMetrics:
    """시장 지표 데이터 클래스"""
    volatility: float = 0.0
    trend_strength: float = 0.0
    momentum: float = 0.0
    rsi: float = 50.0
    liquidity_score: float = 0.0
    volume_profile: float = 0.0
    price_momentum: float = 0.0
    macd: Dict[str, float] = field(default_factory=lambda: {
        'macd': 0.0,
        'signal': 0.0,
        'histogram': 0.0
    })
    
    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            'volatility': float(self.volatility),
            'trend_strength': float(self.trend_strength),
            'momentum': float(self.momentum),
            'rsi': float(self.rsi),
            'liquidity_score': float(self.liquidity_score),
            'volume_profile': float(self.volume_profile),
            'price_momentum': float(self.price_momentum),
            'macd': {k: float(v) for k, v in self.macd.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketMetrics':
        """딕셔너리로부터 객체 생성"""
        metrics = cls()
        for key, value in data.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        return metrics

@dataclass
class MarketState:
    """시장 상태 데이터 클래스"""
    timestamp: datetime
    price: float
    volume: float
    timeframe: str
    metrics: MarketMetrics = field(default_factory=MarketMetrics)
    orderbook: Optional[Dict] = None
    trend: str = "neutral"
    market_condition: str = "normal"
    current_price: float = 0.0    # 현재가 필드 추가
    current_volume: float = 0.0   # 현재 거래량 필드 추가
    last_update: Optional[datetime] = None
    
    def validate(self) -> bool:
        """데이터 유효성 검증"""
        try:
            if not isinstance(self.timestamp, datetime):
                raise ValidationError("timestamp must be datetime object")
            
            if not isinstance(self.price, (int, float)) or self.price <= 0:
                raise ValidationError("Invalid price value")
                
            if not isinstance(self.volume, (int, float)) or self.volume < 0:
                raise ValidationError("Invalid volume value")
                
            if not isinstance(self.timeframe, str) or self.timeframe not in [
                '1m', '5m', '15m', '1h', '4h', '1d'
            ]:
                raise ValidationError("Invalid timeframe")

            if self.current_price <= 0:
                raise ValidationError("Invalid current price")
            
            return True
            
        except Exception as e:
            raise ValidationError(f"Validation error: {str(e)}")
    
    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            'timestamp': self.timestamp,
            'price': float(self.price),
            'volume': float(self.volume),
            'timeframe': self.timeframe,
            'metrics': self.metrics.to_dict(),
            'orderbook': self.orderbook,
            'trend': self.trend,
            'market_condition': self.market_condition,
            'current_price': float(self.current_price),
            'current_volume': float(self.current_volume),
            'last_update': self.last_update,
            'integrated_state': self.get_integrated_state()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketState':
        """딕셔너리로부터 객체 생성"""
        metrics = MarketMetrics.from_dict(data.get('metrics', {}))
        return cls(
            timestamp=data['timestamp'],
            price=data['price'],
            volume=data['volume'],
            timeframe=data['timeframe'],
            metrics=metrics,
            orderbook=data.get('orderbook'),
            trend=data.get('trend', 'neutral'),
            market_condition=data.get('market_condition', 'normal'),
            current_price=data.get('current_price', 0.0),
            current_volume=data.get('current_volume', 0.0),
            last_update=data.get('last_update')
        )

    def calculate_metrics(self, price_history: List[float], volume_history: List[float]) -> None:
        """시장 지표 계산"""
        if len(price_history) < 2:
            return
            
        # 변동성 계산 (유틸리티 함수 사용)
        self.metrics.volatility = calculate_volatility(price_history)
        
        # 추세 강도 계산
        ma_short = np.mean(price_history[-5:])
        ma_long = np.mean(price_history[-20:])
        self.metrics.trend_strength = float((ma_short - ma_long) / ma_long if ma_long != 0 else 0)
        
        # RSI 계산
        self.metrics.rsi = self._calculate_rsi(price_history)
        
        # 모멘텀 계산
        self.metrics.momentum = self._calculate_momentum(price_history)
        
        # 거래량 프로파일 계산
        self.metrics.volume_profile = self._calculate_volume_profile(volume_history)
        
        # MACD 계산
        self.metrics.macd = self._calculate_macd(price_history)
        
        # 현재가 업데이트
        self.current_price = price_history[-1]
        if volume_history:
            self.current_volume = volume_history[-1]
        self.last_update = datetime.now()

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) <= period:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _calculate_momentum(self, prices: List[float], period: int = 10) -> float:
        """모멘텀 계산"""
        if len(prices) <= period:
            return 0.0
            
        return float((prices[-1] - prices[-period]) / prices[-period])

    def _calculate_volume_profile(self, volumes: List[float]) -> float:
        """거래량 프로파일 계산"""
        if not volumes:
            return 0.0
            
        return float(np.mean(volumes[-5:]) / np.mean(volumes))

    def _calculate_macd(
        self,
        prices: List[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, float]:
        """MACD 계산"""
        if len(prices) < slow_period + signal_period:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}

        # 지수이동평균 계산
        ema_fast = self._calculate_ema(prices, fast_period)
        ema_slow = self._calculate_ema(prices, slow_period)
        
        # MACD 라인
        macd_line = float(ema_fast[-1] - ema_slow[-1])
        
        # 시그널 라인
        macd_series = [ema_fast[i] - ema_slow[i] for i in range(len(ema_fast))]
        signal_line = float(self._calculate_ema(macd_series, signal_period)[-1])
        
        # 히스토그램
        histogram = float(macd_line - signal_line)
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def _calculate_ema(self, data: List[float], period: int) -> List[float]:
        """지수이동평균 계산"""
        if not data or len(data) < period:
            return data
            
        multiplier = 2 / (period + 1)
        ema = [data[0]]
        
        for price in data[1:]:
            ema.append(float((price * multiplier) + (ema[-1] * (1 - multiplier))))
            
        return ema

    def get_integrated_state(self) -> Dict:
        """통합 상태 정보"""
        return {
            'overall_trend': self.trend,
            'volatility_state': 'high' if self.metrics.volatility > 0.02 else 'normal',
            'momentum_signal': self.metrics.momentum,
            'market_condition': self.market_condition,
            'updated_at': self.last_update.isoformat() if self.last_update else None
        }