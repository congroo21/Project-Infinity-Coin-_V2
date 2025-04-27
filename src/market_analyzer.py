# src/market_analyzer.py

import logging
import numpy as np
from collections import deque
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from .utils.volatility_utils import calculate_volatility, calculate_returns
import asyncio

from .exceptions import (
    ValidationError, InsufficientDataError, MarketDataError,
    SystemResourceError
)
from .models.market_state import MarketState
from .config import Config

class MultiTimeframeMarketAnalyzer:
    """다중 시간프레임 시장 분석기"""
    def __init__(self):
        # 시간프레임 설정
        self.timeframes = Config.TIMEFRAMES
        self.min_data_points = Config.MIN_DATA_POINTS
        
        # 데이터 저장소 초기화
        self.data_stores = {
            tf: {
                'prices': deque(maxlen=1000),
                'volumes': deque(maxlen=1000),
                'metrics': deque(maxlen=100)
            } for tf in self.timeframes
        }
        
        # 통합 분석 결과 캐시
        self.cached_analysis = None
        self.last_cache_update = None
        self.cache_duration = timedelta(milliseconds=200)  # 200ms
        
        # 실행 상태
        self.running = False
        self.initialization_done = False
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)

    async def start(self):
        """분석기 시작"""
        try:
            self.running = True
            self.logger.info("멀티 타임프레임 시장 분석기 시작")
            
            # 초기 데이터 검증
            if not self._validate_initial_data():
                raise InsufficientDataError("초기 데이터가 부족합니다")
            
            # 분석 태스크 실행
            analysis_task = asyncio.create_task(self._analysis_loop())
            
            # 초기화 완료
            self.initialization_done = True
            
            await analysis_task
            
        except Exception as e:
            self.logger.error(f"시장 분석기 시작 오류: {e}")
            self.running = False
            raise

    def _validate_initial_data(self) -> bool:
        """초기 데이터 유효성 검증"""
        for tf in self.timeframes:
            required_points = self.min_data_points.get(tf, 30)
            store = self.data_stores[tf]
            
            if (len(store['prices']) < required_points or
                len(store['volumes']) < required_points):
                self.logger.warning(f"{tf} 시간프레임의 데이터가 부족합니다")
                return False
        return True

    async def _analysis_loop(self):
        """메인 분석 루프"""
        while self.running:
            try:
                # 각 시간프레임별 분석
                analysis_results = {}
                for tf in self.timeframes:
                    try:
                        result = await self._analyze_timeframe(tf)
                        if result:
                            analysis_results[tf] = result
                    except Exception as e:
                        self.logger.error(f"{tf} 분석 오류: {e}")
                        continue

                # 통합 분석 수행
                if analysis_results:
                    integrated_result = self._integrate_analysis(analysis_results)
                    if integrated_result:
                        self.cached_analysis = integrated_result
                        self.last_cache_update = datetime.now()

                # 시스템 리소스 체크
                if not self._check_system_resources():
                    raise SystemResourceError("시스템 리소스 부족")

                # 분석 간격 대기
                await asyncio.sleep(0.001)  # 1ms

            except Exception as e:
                self.logger.error(f"분석 루프 오류: {e}")
                await asyncio.sleep(1)

    def _check_system_resources(self) -> bool:
        """시스템 리소스 사용량 체크"""
        try:
            import psutil
            # CPU 사용량 체크
            cpu_usage = psutil.cpu_percent()
            if cpu_usage > Config.SYSTEM_METRICS['cpu_threshold']:
                self.logger.warning(f"높은 CPU 사용량: {cpu_usage}%")
                return False

            # 메모리 사용량 체크
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > Config.SYSTEM_METRICS['memory_threshold']:
                self.logger.warning(f"높은 메모리 사용량: {memory_usage}%")
                return False

            return True
            
        except Exception as e:
            self.logger.error(f"리소스 체크 오류: {e}")
            return True  # 체크 실패 시 기본값
        
    async def _analyze_timeframe(self, timeframe: str) -> Optional[Dict]:
        """각 시간프레임별 분석 수행"""
        try:
            # 캐시 확인
            cache_key = f"{timeframe}_analysis"
            if hasattr(self, '_analysis_cache') and cache_key in self._analysis_cache:
                if datetime.now() - self._analysis_cache[cache_key]['timestamp'] < timedelta(seconds=10):
                    return self._analysis_cache[cache_key]['data']

            store = self.data_stores[timeframe]
            if not self._validate_data_store(store, timeframe):
                return None

            # 가격 데이터 분석
            price_metrics = self._analyze_price_action(
                list(store['prices']),
                list(store['volumes'])
            )
            if not price_metrics:
                return None

            # 추세 분석
            trend_analysis = self._analyze_trend(list(store['prices']))
            
            # 변동성 분석
            volatility = self._calculate_volatility(list(store['prices']))
            
            # 모멘텀 분석
            momentum = self._calculate_momentum(list(store['prices']))
            
            # 거래량 분석
            volume_analysis = self._analyze_volume(list(store['volumes']))

            analysis_result = {
                'timestamp': datetime.now(),
                'timeframe': timeframe,
                'trend': trend_analysis['trend'],
                'trend_strength': trend_analysis['strength'],
                'volatility': volatility,
                'momentum': momentum,
                'volume_profile': volume_analysis,
                'price_metrics': price_metrics,
                'overall_trend': trend_analysis['overall_trend']
            }

            # 메트릭스 저장
            store['metrics'].append(analysis_result)
            
            # 계산 결과 캐싱
            if not hasattr(self, '_analysis_cache'):
                self._analysis_cache = {}
                
            self._analysis_cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': analysis_result
            }
            
            return analysis_result

        except Exception as e:
            self.logger.error(f"{timeframe} 분석 오류: {e}")
            return None

    def _validate_data_store(self, store: Dict, timeframe: str) -> bool:
        """데이터 저장소 유효성 검증"""
        min_required = self.min_data_points.get(timeframe, 30)
        return (len(store['prices']) >= min_required and
                len(store['volumes']) >= min_required)

    def _analyze_price_action(
        self,
        prices: List[float],
        volumes: List[float]
    ) -> Optional[Dict]:
        """가격 행동 분석"""
        try:
            if len(prices) < 2:
                return None
                
            # 기본 가격 지표
            current_price = prices[-1]
            price_change = (current_price - prices[-2]) / prices[-2]
            
            # 이동평균 계산
            ma_5 = np.mean(prices[-5:])
            ma_10 = np.mean(prices[-10:])
            ma_20 = np.mean(prices[-20:])
            
            # MACD 계산
            macd = self._calculate_macd(prices)
            
            # RSI 계산
            rsi = self._calculate_rsi(prices)

            return {
                'current_price': float(current_price),
                'price_change': float(price_change),
                'ma_5': float(ma_5),
                'ma_10': float(ma_10),
                'ma_20': float(ma_20),
                'macd': macd,
                'rsi': rsi
            }

        except Exception as e:
            self.logger.error(f"가격 분석 오류: {e}")
            return None

    def _analyze_trend(self, prices: List[float]) -> Dict:
        """추세 분석"""
        try:
            if len(prices) < 20:
                return {'trend': 'neutral', 'strength': 0.0, 'overall_trend': 'neutral'}

            # 단기/중기/장기 이동평균
            ma_short = np.mean(prices[-5:])
            ma_mid = np.mean(prices[-10:])
            ma_long = np.mean(prices[-20:])

            # 추세 강도 계산
            short_strength = (ma_short - ma_mid) / ma_mid
            long_strength = (ma_mid - ma_long) / ma_long
            
            # 전체 강도
            total_strength = (short_strength + long_strength) / 2

            # 추세 판단
            if total_strength > 0.01:
                trend = 'uptrend'
                if total_strength > 0.03:
                    overall_trend = 'strong_uptrend'
                else:
                    overall_trend = 'uptrend'
            elif total_strength < -0.01:
                trend = 'downtrend'
                if total_strength < -0.03:
                    overall_trend = 'strong_downtrend'
                else:
                    overall_trend = 'downtrend'
            else:
                trend = 'neutral'
                overall_trend = 'neutral'

            return {
                'trend': trend,
                'strength': float(abs(total_strength)),
                'overall_trend': overall_trend
            }

        except Exception as e:
            self.logger.error(f"추세 분석 오류: {e}")
            return {'trend': 'neutral', 'strength': 0.0, 'overall_trend': 'neutral'}

    def _calculate_momentum(self, prices: List[float]) -> float:
        """모멘텀 계산"""
        try:
            if len(prices) < 10:
                return 0.0

            # Rate of Change (ROC)
            roc = (prices[-1] - prices[-10]) / prices[-10]
            
            # 정규화 (-1 ~ 1 범위로)
            normalized_momentum = np.tanh(roc * 10)
            
            return float(normalized_momentum)

        except Exception as e:
            self.logger.error(f"모멘텀 계산 오류: {e}")
            return 0.0

    def _analyze_volume(self, volumes: List[float]) -> Dict:
        """거래량 분석"""
        try:
            if not volumes:
                return {'profile': 'normal', 'intensity': 0.0}

            # 평균 거래량 계산
            avg_volume = np.mean(volumes)
            recent_volume = np.mean(volumes[-5:])
            
            # 거래량 강도
            intensity = (recent_volume - avg_volume) / avg_volume
            
            # 거래량 프로파일 판단
            if intensity > 0.5:
                profile = 'high'
            elif intensity < -0.5:
                profile = 'low'
            else:
                profile = 'normal'

            return {
                'profile': profile,
                'intensity': float(intensity)
            }

        except Exception as e:
            self.logger.error(f"거래량 분석 오류: {e}")
            return {'profile': 'normal', 'intensity': 0.0}

    def _calculate_macd(
        self,
        prices: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict:
        """MACD 계산"""
        try:
            if len(prices) < slow + signal:
                return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}

            # 지수이동평균 계산
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            
            # MACD 라인
            macd_line = ema_fast - ema_slow
            
            # 시그널 라인
            signal_line = self._calculate_ema(macd_line, signal)
            
            # 히스토그램
            histogram = macd_line - signal_line

            return {
                'macd': float(macd_line[-1]),
                'signal': float(signal_line[-1]),
                'histogram': float(histogram[-1])
            }

        except Exception as e:
            self.logger.error(f"MACD 계산 오류: {e}")
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
    def _calculate_ema(self, data: List[float], period: int) -> np.ndarray:
        """지수이동평균 계산"""
        try:
            if len(data) < period:
                return np.array(data)

            alpha = 2 / (period + 1)
            ema = np.zeros_like(data)
            ema[0] = data[0]
            
            for i in range(1, len(data)):
                ema[i] = data[i] * alpha + ema[i-1] * (1 - alpha)
                
            return ema

        except Exception as e:
            self.logger.error(f"EMA 계산 오류: {e}")
            return np.array(data)

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI 계산"""
        try:
            if len(prices) < period + 1:
                return 50.0

            deltas = np.diff(prices)
            seed = deltas[:period+1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            
            if down == 0:
                return 100.0
                
            rs = up / down
            return float(100 - (100 / (1 + rs)))

        except Exception as e:
            self.logger.error(f"RSI 계산 오류: {e}")
            return 50.0

    async def get_integrated_analysis(self) -> Dict:
        """전체 시간프레임 통합 분석"""
        try:
            # 캐시 확인
            if self._is_cache_valid():
                return self.cached_analysis

            # 각 시간프레임의 최근 분석 결과 수집
            timeframe_results = {}
            for tf in self.timeframes:
                if tf not in self.data_stores:
                    continue
                    
                metrics = self.data_stores[tf].get('metrics', deque())
                if metrics:
                    timeframe_results[tf] = metrics[-1]

            if not timeframe_results:
                raise InsufficientDataError("분석 결과 없음")

            # 통합 분석 수행
            integrated_signals = {
                'timestamp': datetime.now(),
                'overall_trend': self._determine_overall_trend(timeframe_results),
                'volatility_state': self._assess_volatility_state(timeframe_results),
                'momentum_signal': self._calculate_momentum_signal(timeframe_results),
                'trading_signals': self._generate_trading_signals(timeframe_results),
                'risk_metrics': self._calculate_risk_metrics(timeframe_results),
                'market_condition': self._assess_market_condition(timeframe_results)
            }

            # 캐시 업데이트
            self.cached_analysis = integrated_signals
            self.last_cache_update = datetime.now()

            return integrated_signals

        except Exception as e:
            self.logger.error(f"통합 분석 오류: {e}")
            # 캐시된 결과라도 반환
            return self.cached_analysis if self.cached_analysis else {}

    def _is_cache_valid(self) -> bool:
        """캐시 유효성 확인"""
        if not self.last_cache_update:
            return False
        return datetime.now() - self.last_cache_update < self.cache_duration

    def _determine_overall_trend(self, results: Dict[str, Dict]) -> str:
        """전체 추세 판단"""
        try:
            # 시간프레임별 가중치
            weights = {
                '1m': 0.1,
                '5m': 0.2,
                '15m': 0.2,
                '1h': 0.2,
                '4h': 0.15,
                '1d': 0.15
            }
            
            trend_scores = []
            total_weight = 0
            
            for tf, result in results.items():
                if tf not in weights:
                    continue
                    
                # 추세 점수 계산
                if result.get('overall_trend') == 'strong_uptrend':
                    score = 2
                elif result.get('overall_trend') == 'uptrend':
                    score = 1
                elif result.get('overall_trend') == 'strong_downtrend':
                    score = -2
                elif result.get('overall_trend') == 'downtrend':
                    score = -1
                else:
                    score = 0
                    
                weight = weights[tf]
                trend_scores.append(score * weight)
                total_weight += weight

            if not trend_scores:
                return "neutral"

            # 가중 평균 계산
            weighted_score = sum(trend_scores) / total_weight
            
            # 최종 추세 판단
            if weighted_score > 1.5:
                return "strong_uptrend"
            elif weighted_score > 0.5:
                return "uptrend"
            elif weighted_score < -1.5:
                return "strong_downtrend"
            elif weighted_score < -0.5:
                return "downtrend"
            return "neutral"

        except Exception as e:
            self.logger.error(f"추세 판단 오류: {e}")
            return "neutral"

    def _assess_volatility_state(self, results: Dict[str, Dict]) -> str:
        """변동성 상태 평가"""
        try:
            volatilities = [
                r.get('volatility', 0.0)
                for r in results.values()
            ]
            
            if not volatilities:
                return "normal"

            avg_volatility = np.mean(volatilities)
            
            if avg_volatility > 0.03:  # 3% 이상
                return "high"
            elif avg_volatility < 0.01:  # 1% 미만
                return "low"
            return "normal"

        except Exception as e:
            self.logger.error(f"변동성 평가 오류: {e}")
            return "normal"

    def _calculate_momentum_signal(self, results: Dict[str, Dict]) -> float:
        """통합 모멘텀 시그널 계산"""
        try:
            momentums = [
                r.get('momentum', 0.0)
                for r in results.values()
            ]
            
            if not momentums:
                return 0.0

            # 지수가중이동평균 사용
            weights = np.exp(np.linspace(-1, 0, len(momentums)))
            weights /= weights.sum()
            
            weighted_momentum = np.sum(
                np.array(momentums) * weights
            )
            
            return float(weighted_momentum)

        except Exception as e:
            self.logger.error(f"모멘텀 시그널 계산 오류: {e}")
            return 0.0

    def _generate_trading_signals(self, results: Dict[str, Dict]) -> Dict:
        """트레이딩 신호 생성"""
        try:
            signals = {
                'entry_points': [],
                'exit_points': []
            }

            for tf, result in results.items():
                if self._is_entry_condition(result):
                    direction = 'long' if result.get('trend') == 'uptrend' else 'short'
                    confidence = min(
                        abs(result.get('momentum', 0)),
                        result.get('trend_strength', 0)
                    )
                    signals['entry_points'].append((direction, confidence))
                    
                if self._is_exit_condition(result):
                    signals['exit_points'].append(tf)

            return signals

        except Exception as e:
            self.logger.error(f"신호 생성 오류: {e}")
            return {'entry_points': [], 'exit_points': []}

    def _is_entry_condition(self, result: Dict) -> bool:
        """진입 조건 확인"""
        try:
            trend = result.get('trend', 'neutral')
            momentum = abs(result.get('momentum', 0))
            volatility = result.get('volatility', 0)
            
            return (
                trend in ['uptrend', 'downtrend'] and
                momentum > 0.5 and
                volatility < 0.03
            )

        except Exception as e:
            self.logger.error(f"진입 조건 확인 오류: {e}")
            return False

    def _is_exit_condition(self, result: Dict) -> bool:
        """청산 조건 확인"""
        try:
            trend_change = result.get('trend') != result.get('previous_trend')
            high_volatility = result.get('volatility', 0) > 0.03
            momentum_weak = abs(result.get('momentum', 0)) < 0.2
            
            return trend_change or high_volatility or momentum_weak

        except Exception as e:
            self.logger.error(f"청산 조건 확인 오류: {e}")
            return False

    # src/market_analyzer.py의 리스크 계산 관련 함수 수정

    def _calculate_volatility(self, prices: List[float]) -> float:
        """변동성 계산"""
        from src.utils.risk_utils import risk_calculator
        returns = risk_calculator.calculate_returns(prices)
        return risk_calculator.calculate_volatility(returns)

    def _calculate_risk_metrics(self, timeframe_results: Dict) -> Dict:
        """리스크 지표 계산"""
        try:
            # 중앙화된 리스크 계산기 사용
            from src.utils.risk_utils import risk_calculator
            
            # 여러 시간프레임의 변동성 수집
            volatilities = [r.get('volatility', 0) for r in timeframe_results.values()]
            
            # 각 시간프레임의 수익률 추출
            all_returns = []
            for result in timeframe_results.values():
                if 'returns' in result:
                    all_returns.extend(result['returns'])
            
            if all_returns:
                # 통합 리스크 메트릭스 계산
                integrated_metrics = risk_calculator.calculate_risk_metrics(all_returns)
                
                # 추가 정보
                integrated_metrics.update({
                    'volatility_level': float(np.mean(volatilities)),
                    'trend_reliability': self._calculate_trend_reliability(timeframe_results),
                    'market_risk': self._calculate_market_risk(timeframe_results)
                })
                
                return integrated_metrics
            else:
                return {
                    'volatility_level': float(np.mean(volatilities)),
                    'trend_reliability': self._calculate_trend_reliability(timeframe_results),
                    'market_risk': self._calculate_market_risk(timeframe_results)
                }

        except Exception as e:
            self.logger.error(f"리스크 지표 계산 오류: {e}")
            return {
                'volatility_level': 0.0,
                'trend_reliability': 0.0,
                'market_risk': 0.5
            }

    def _calculate_market_risk(self, results: Dict[str, Dict]) -> float:
        """시장 리스크 점수 계산"""
        try:
            risk_factors = []
            
            for result in results.values():
                volatility = result.get('volatility', 0)
                momentum = abs(result.get('momentum', 0))
                volume_intensity = result.get('volume_profile', {}).get('intensity', 0)
                
                risk_score = (
                    volatility * 0.4 +
                    momentum * 0.3 +
                    abs(volume_intensity) * 0.3
                )
                risk_factors.append(risk_score)
            
            return float(np.mean(risk_factors)) if risk_factors else 0.5

        except Exception as e:
            self.logger.error(f"시장 리스크 계산 오류: {e}")
            return 0.5

    def _assess_market_condition(self, results: Dict[str, Dict]) -> str:
        """시장 상태 평가"""
        try:
            overall_trend = self._determine_overall_trend(results)
            volatility = self._assess_volatility_state(results)
            
            if overall_trend in ['strong_uptrend', 'strong_downtrend'] and volatility == 'high':
                return 'volatile_trending'
            elif overall_trend in ['uptrend', 'downtrend'] and volatility == 'normal':
                return 'stable_trending'
            elif overall_trend == 'neutral' and volatility == 'low':
                return 'ranging'
            elif volatility == 'high':
                return 'volatile'
            return 'normal'

        except Exception as e:
            self.logger.error(f"시장 상태 평가 오류: {e}")
            return 'normal'
        
    def _integrate_analysis(self, timeframe_results: Dict) -> Dict:
        """통합 시장 분석 수행"""
        try:
            # 입력 데이터 검증
            if not timeframe_results:
                raise InsufficientDataError("분석 결과 없음")

            # 통합 분석 결과 생성
            integrated_signals = {
                'timestamp': datetime.now(),
                'overall_trend': self._determine_overall_trend(timeframe_results),
                'volatility_state': self._assess_volatility_state(timeframe_results),
                'momentum_signal': self._calculate_momentum_signal(timeframe_results),
                'trading_signals': self._generate_trading_signals(timeframe_results),
                'risk_metrics': self._calculate_risk_metrics(timeframe_results),
                'market_condition': self._assess_market_condition(timeframe_results)
            }

            return integrated_signals
            
        except Exception as e:
            self.logger.error(f"통합 분석 오류: {e}")
            # 캐시된 결과라도 반환
            return self.cached_analysis if self.cached_analysis else {}