# src/scenarios/trading/trading_system.py

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple  # Tuple 추가
from dataclasses import dataclass
import numpy as np

from ...models.market_state import MarketState, MarketMetrics
from ..risk.risk_manager import ImprovedRiskManager
from ...database import ImprovedDatabaseManager
from ...exceptions import (
    TradeExecutionError, ValidationError, RiskLimitExceededError,
    InsufficientDataError, EmergencyShutdownError, SystemEmergencyError,
    ConfigurationError  # ConfigurationError 추가
)
from ...config import Config

@dataclass
class TradeSignal:
    """거래 신호 데이터 클래스"""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    price: float
    size: float
    timeframe: str
    confidence: float
    signal_source: str
    risk_score: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def validate(self) -> bool:
        """거래 신호 유효성 검증"""
        try:
            if not isinstance(self.timestamp, datetime):
                return False
            if not isinstance(self.price, (int, float)) or self.price <= 0:
                return False
            if not isinstance(self.size, float) or self.size <= 0:
                return False
            if self.action not in ['buy', 'sell', 'hold', 'close']:  # 'close' 추가
                return False
            if not isinstance(self.confidence, float) or not 0 <= self.confidence <= 1:
                return False
            # stop_loss와 take_profit 검증 추가
            if self.stop_loss is not None and self.stop_loss <= 0:
                return False
            if self.take_profit is not None and self.take_profit <= 0:
                return False
            if self.risk_score < 0 or self.risk_score > 1:
                return False
            return True
        except Exception as e:
            logging.error(f"Signal validation error: {e}")
            return False

@dataclass
class ExecutionResult:
    """거래 실행 결과 데이터 클래스"""
    timestamp: datetime
    success: bool
    trade_id: Optional[str]
    error_message: Optional[str]
    execution_price: Optional[float]
    filled_size: Optional[float]
    fees: Optional[float]
    latency: float  # 실행 소요 시간 (초)
    pnl: Optional[float] = None  # 실현 손익 추가

class ImprovedTradingSystem:
    """개선된 트레이딩 시스템"""
    def __init__(self):
        # 기본 컴포넌트 초기화
        self.risk_manager = ImprovedRiskManager()
        self.db_manager = ImprovedDatabaseManager()
        
        # 실행 상태 및 설정
        self.running = False
        self.emergency_mode = False
        self.current_positions = {}
        self.trade_history = []
        self.last_trade_time = None
        self.consecutive_errors = 0
        self.error_cooldown = False
        self.cooldown_until = None
        
        # 성능 모니터링
        self.execution_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit_loss': 0.0,
            'average_latency': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # 설정 로드
        self.load_config()
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 시장 상태 캐시
        self.market_state_cache = {}
        self.last_cache_update = None
        self.cache_duration = timedelta(milliseconds=50)  # 50ms 캐시 유지

    def load_config(self):
        """설정 로드 및 검증"""
        try:
            self.trade_interval = Config.TRADE_INTERVAL
            self.max_position_size = Config.MAX_POSITION_SIZE
            self.emergency_stop_loss = Config.EMERGENCY_STOP_LOSS
            
            # 설정 유효성 검증
            if self.trade_interval <= 0:
                raise ValidationError("Invalid trade interval")
            if self.max_position_size <= 0 or self.max_position_size > 1:
                raise ValidationError("Invalid position size limit")
            if self.emergency_stop_loss <= 0 or self.emergency_stop_loss > 0.5:
                raise ValidationError("Invalid emergency stop loss")
                
        except Exception as e:
            self.logger.error(f"Configuration error: {e}")
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")

    def _is_cache_valid(self) -> bool:
        """캐시 유효성 확인"""
        if not self.last_cache_update:
            return False
        return datetime.now() - self.last_cache_update < self.cache_duration
    
    async def _generate_trading_signal(self, market_data: Dict) -> Optional[TradeSignal]:
        """거래 신호 생성"""
        try:
            if not self._validate_market_data(market_data):
                raise ValidationError("Invalid market data")

            # 시장 분석 수행
            analysis_result = await self._analyze_market(market_data)
            if not analysis_result:
                return None

            # 신호 생성 조건 확인
            trend = analysis_result['overall_trend']
            momentum = float(analysis_result['momentum_signal'])
            volatility = analysis_result.get('volatility_state', 'normal')
            
            # 행동 결정
            action = 'hold'
            confidence = 0.0
            size = 0.0
            
            if abs(momentum) >= 0.7 and volatility != 'high':  # 강한 모멘텀, 낮은 변동성
                if trend in ['strong_uptrend', 'uptrend'] and momentum > 0:
                    action = 'buy'
                    confidence = min(abs(momentum), 0.9)  # 최대 90% 신뢰도
                    size = self._calculate_position_size(confidence)
                elif trend in ['strong_downtrend', 'downtrend'] and momentum < 0:
                    action = 'sell'
                    confidence = min(abs(momentum), 0.9)
                    size = self._calculate_position_size(confidence)

            # 리스크 스코어 계산
            risk_score = self._calculate_risk_score(analysis_result)

            # 손절가/익절가 계산
            stop_loss, take_profit = self._calculate_price_levels(
                market_data['price'],
                action,
                risk_score
            )

            return TradeSignal(
                timestamp=datetime.now(),
                symbol=market_data.get('symbol', 'DEFAULT'),
                action=action,
                price=float(market_data['price']),
                size=size,
                timeframe=market_data.get('timeframe', '1m'),
                confidence=confidence,
                signal_source='market_analysis',
                risk_score=risk_score,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return None

    def _validate_market_data(self, data: Dict) -> bool:
        """시장 데이터 유효성 검증"""
        required_fields = ['price', 'timestamp', 'integrated_state']
        return all(field in data for field in required_fields)

    async def _analyze_market(self, market_data: Dict) -> Optional[Dict]:
        """시장 분석 수행"""
        try:
            # 캐시 확인
            if self._is_cache_valid():
                return self.market_state_cache.get('analysis')

            # 통합 상태 분석
            integrated_state = market_data.get('integrated_state', {})
            
            # 추가 지표 계산
            volatility = self._calculate_volatility(market_data)
            momentum = self._calculate_momentum(market_data)
            
            # 여기서 딕셔너리 형식으로 접근하던 것을 속성 접근으로 변경
            analysis_result = {
                'timestamp': datetime.now(),
                'overall_trend': integrated_state.get('overall_trend', 'neutral'),
                'volatility_state': 'high' if volatility > self.risk_manager.thresholds.volatility_threshold else 'normal',
                'momentum_signal': momentum,
                'market_condition': integrated_state.get('market_condition', 'normal')
            }

            # 캐시 업데이트
            self.market_state_cache['analysis'] = analysis_result
            self.last_cache_update = datetime.now()

            return analysis_result

        except Exception as e:
            self.logger.error(f"Market analysis error: {e}")
            return None

    def _calculate_position_size(self, confidence: float) -> float:
        """포지션 크기 계산"""
        base_size = self.max_position_size * confidence
        return min(base_size, self.max_position_size)

    def _calculate_risk_score(self, analysis: Dict) -> float:
        """리스크 점수 계산"""
        try:
            volatility_weight = 1.5 if analysis.get('volatility_state') == 'high' else 1.0
            trend_strength = abs(float(analysis.get('momentum_signal', 0)))
            market_condition_weight = 1.2 if analysis.get('market_condition') == 'volatile' else 1.0
            
            risk_score = (
                (1 - trend_strength) * 0.4 +
                volatility_weight * 0.4 +
                market_condition_weight * 0.2
            )
            return min(max(risk_score, 0), 1)  # 0~1 범위로 제한
            
        except Exception as e:
            self.logger.error(f"Risk score calculation error: {e}")
            return 0.8  # 기본값으로 높은 리스크 점수 반환

    def _calculate_price_levels(
        self,
        current_price: float,
        action: str,
        risk_score: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """손절가/익절가 계산"""
        if action == 'hold':
            return None, None

        # 리스크 점수에 따른 손절폭 조정
        stop_loss_pct = self.emergency_stop_loss * (1 + risk_score)  # 리스크가 높을수록 더 타이트한 손절
        take_profit_pct = stop_loss_pct * 2  # 2배 수익 목표

        if action == 'buy':
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        else:  # sell
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)

        return stop_loss, take_profit

    def _calculate_volatility(self, market_data: Dict) -> float:
        """변동성 계산"""
        try:
            if 'integrated_state' in market_data:
                return float(market_data['integrated_state'].get('volatility', 0.0))
            return 0.0
        except Exception:
            return 0.0

    def _calculate_momentum(self, market_data: Dict) -> float:
        """모멘텀 계산"""
        try:
            if 'integrated_state' in market_data:
                return float(market_data['integrated_state'].get('momentum_signal', 0.0))
            return 0.0
        except Exception:
            return 0.0

    async def _calculate_performance_metrics(self) -> Dict:
        """성능 지표 계산"""
        try:
            if not self.trade_history:
                return {}

            recent_trades = self.trade_history[-100:]  # 최근 100개 거래
            
            # 수익률 계산
            profits = [trade.get('pnl', 0) for trade in recent_trades]
            total_profit = sum(profits)
            win_count = sum(1 for p in profits if p > 0)
            
            # 승률
            win_rate = win_count / len(recent_trades) if recent_trades else 0
            
            # 최대 낙폭
            cumulative_profits = np.cumsum(profits)
            max_drawdown = self._calculate_max_drawdown(cumulative_profits)
            
            # 샤프 비율
            returns = np.array(profits)
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            else:
                sharpe_ratio = 0

            # 결과 업데이트
            self.execution_stats.update({
                'total_profit_loss': total_profit,
                'win_rate': win_rate * 100,
                'max_drawdown': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio
            })

            return self.execution_stats

        except Exception as e:
            self.logger.error(f"Performance metrics calculation error: {e}")
            return self.execution_stats

    def _calculate_max_drawdown(self, cumulative_profits: np.ndarray) -> float:
        """최대 낙폭 계산"""
        running_max = np.maximum.accumulate(cumulative_profits)
        drawdowns = (running_max - cumulative_profits) / (running_max + 1e-10)
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    async def _check_portfolio_risk(self):
        """포트폴리오 전체 리스크 체크"""
        try:
            if not self.current_positions:
                return

            # 전체 포트폴리오 가치 계산
            total_value = await self._calculate_portfolio_value()
            
            # 포지션 집중도 체크
            concentration = await self._check_position_concentration()
            
            # 상관관계 체크
            correlation = await self._check_position_correlation()
            
            # 변동성 체크
            volatility = await self._check_portfolio_volatility()

            # 종합 리스크 평가 - 임계값 속성 접근 방식으로 수정
            if (correlation > self.risk_manager.thresholds.correlation_threshold or  # 속성으로 접근
                concentration > self.risk_manager.thresholds.concentration_limit or  # 속성으로 접근
                volatility > self.risk_manager.thresholds.volatility_threshold):     # 속성으로 접근
                
                await self._reduce_portfolio_risk()

        except Exception as e:
            self.logger.error(f"Portfolio risk check error: {e}")

    async def _calculate_portfolio_value(self) -> float:
        """포트폴리오 총 가치 계산"""
        try:
            total_value = 0.0
            for symbol, position in self.current_positions.items():
                current_price = await self._get_current_price(symbol)
                if current_price:
                    total_value += position['size'] * current_price
            return total_value
        except Exception as e:
            self.logger.error(f"Portfolio value calculation error: {e}")
            return 0.0

    async def _check_position_concentration(self) -> float:
        """포지션 집중도 체크"""
        try:
            total_value = await self._calculate_portfolio_value()
            if not total_value:
                return 0.0

            # Herfindahl-Hirschman Index 계산
            position_weights = []
            for symbol, position in self.current_positions.items():
                current_price = await self._get_current_price(symbol)
                if current_price:
                    weight = (position['size'] * current_price) / total_value
                    position_weights.append(weight)

            return sum(w * w for w in position_weights)

        except Exception as e:
            self.logger.error(f"Position concentration check error: {e}")
            return 0.0

    async def _check_position_correlation(self) -> float:
        """포지션 간 상관관계 체크"""
        try:
            if len(self.current_positions) < 2:
                return 0.0

            # 가격 이력 수집
            price_history = {}
            for symbol in self.current_positions.keys():
                prices = await self._get_price_history(symbol)
                if prices:
                    price_history[symbol] = prices

            if len(price_history) < 2:
                return 0.0

            # 상관계수 행렬 계산
            correlations = []
            symbols = list(price_history.keys())
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    if len(price_history[symbols[i]]) == len(price_history[symbols[j]]):
                        corr = np.corrcoef(
                            price_history[symbols[i]],
                            price_history[symbols[j]]
                        )[0, 1]
                        correlations.append(abs(corr))

            return max(correlations) if correlations else 0.0

        except Exception as e:
            self.logger.error(f"Position correlation check error: {e}")
            return 0.0

    async def _check_portfolio_volatility(self) -> float:
        """포트폴리오 전체 변동성 체크"""
        try:
            if not self.trade_history:
                return 0.0

            # 최근 100개 거래의 수익률
            recent_returns = [
                trade.get('pnl', 0) / trade.get('price', 1)
                for trade in self.trade_history[-100:]
            ]

            return float(np.std(recent_returns) * np.sqrt(252))

        except Exception as e:
            self.logger.error(f"Portfolio volatility check error: {e}")
            return 0.0

    async def _reduce_portfolio_risk(self):
        """포트폴리오 리스크 감소 조치"""
        try:
            # 가장 위험한 포지션 식별
            risky_positions = await self._identify_risky_positions()
            
            # 리스크 감소 조치
            for position in risky_positions:
                if position['risk_level'] == 'high':
                    # 전체 청산
                    await self._close_position(position['symbol'], "risk_reduction")
                elif position['risk_level'] == 'medium':
                    # 부분 청산
                    await self._reduce_position(
                        position['symbol'],
                        position['size'] * 0.5
                    )

        except Exception as e:
            self.logger.error(f"Portfolio risk reduction error: {e}")

    async def _identify_risky_positions(self) -> List[Dict]:
        """위험한 포지션 식별"""
        try:
            risky_positions = []
            for symbol, position in self.current_positions.items():
                current_price = await self._get_current_price(symbol)
                if not current_price:
                    continue

                # 손실 계산
                entry_price = position['entry_price']
                loss_pct = (current_price - entry_price) / entry_price
                if position['action'] == 'sell':
                    loss_pct = -loss_pct

                # 리스크 레벨 결정
                risk_level = 'low'
                if loss_pct < -self.emergency_stop_loss * 0.7:
                    risk_level = 'high'
                elif loss_pct < -self.emergency_stop_loss * 0.5:
                    risk_level = 'medium'

                risky_positions.append({
                    'symbol': symbol,
                    'risk_level': risk_level,
                    'loss_pct': loss_pct,
                    'size': position['size']
                })

            return sorted(
                risky_positions,
                key=lambda x: abs(x['loss_pct']),
                reverse=True
            )

        except Exception as e:
            self.logger.error(f"Risk position identification error: {e}")
            return []

    async def _reduce_position(self, symbol: str, reduction_size: float):
        """포지션 크기 감소"""
        try:
            position = self.current_positions[symbol]
            if position['size'] <= reduction_size:
                await self._close_position(symbol, "full_reduction")
                return

            # 감소 신호 생성
            reduce_signal = TradeSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action='sell' if position['action'] == 'buy' else 'buy',
                price=await self._get_current_price(symbol),
                size=reduction_size,
                timeframe='1m',
                confidence=1.0,
                signal_source='risk_reduction',
                risk_score=0.0
            )

            # 거래 실행
            result = await self._execute_trade(reduce_signal)
            if result.success:
                position['size'] -= reduction_size
                self.logger.info(
                    f"Reduced position for {symbol} by {reduction_size}"
                )

        except Exception as e:
            self.logger.error(f"Position reduction error: {e}")

    async def _handle_emergency_mode(self):
        """긴급 상황 처리"""
        try:
            self.logger.warning("Handling emergency mode...")
            
            # 모든 포지션 청산
            await self._close_all_positions()
            
            # 거래 중지
            self.running = False
            
            # 긴급 상황 기록
            await self.db_manager.save_risk_event({
                'timestamp': datetime.now(),
                'event_type': 'emergency',
                'severity': 'critical',
                'description': 'Emergency shutdown initiated',
                'action_taken': 'All positions closed'
            })
            
            raise SystemEmergencyError("Trading system emergency shutdown")
            
        except Exception as e:
            self.logger.error(f"Emergency handling error: {e}")
            raise

    async def _get_price_history(self, symbol: str) -> List[float]:
        """가격 이력 조회"""
        # 실제 구현 시 데이터베이스나 거래소 API에서 조회
        return [50000000 + i * 1000 for i in range(100)]  # 임시 데이터

    async def _close_position(self, symbol: str, reason: str):
        """포지션 청산"""
        # 여기에 청산 로직 구현 (실제 거래 실행)
        if symbol in self.current_positions:
            self.logger.info(f"Closing position for {symbol} due to {reason}")
            del self.current_positions[symbol]
            return True
        return False

    async def _close_all_positions(self):
        """모든 포지션 청산"""
        for symbol in list(self.current_positions.keys()):
            await self._close_position(symbol, "emergency_closure")
        self.current_positions.clear()

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """현재 가격 조회"""
        # 임시 구현 (실제로는 외부 API 사용)
        return 50000000  # 예시 가격

    async def _execute_trade(self, signal: TradeSignal) -> ExecutionResult:
        """거래 실행"""
        # 여기에 실제 거래 실행 로직 구현
        # 임시 성공 응답
        return ExecutionResult(
            timestamp=datetime.now(),
            success=True,
            trade_id="test_id",
            error_message=None,
            execution_price=signal.price,
            filled_size=signal.size,
            fees=signal.price * signal.size * 0.0005,  # 0.05% 수수료 가정
            latency=0.01,  # 10ms 실행 시간 가정
            pnl=0.0  # 신규 거래는 손익 없음
        )

    def _check_emergency_conditions(self, risk_metrics: Dict) -> bool:
        """긴급 상황 조건 체크 (기존 로직)"""
        # 임시 구현: 변동성이 매우 높으면 긴급 상황으로 간주
        # 여기서 속성 접근 방식으로 수정
        return risk_metrics.get('volatility', 0) > (self.risk_manager.thresholds.volatility_threshold * 2)