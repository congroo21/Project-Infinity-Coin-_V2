# src/news_trading_signal.py

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

@dataclass
class TradingSignal:
    """트레이딩 신호"""
    timestamp: datetime
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    position_size: float
    target_price: float
    stop_loss: float
    time_horizon: int  # 초 단위
    signal_source: str
    additional_info: Dict


class NewsSignalGenerator:
    """뉴스 기반 트레이딩 신호 생성기"""
    def __init__(self, config: Dict):
        self.config = config
        self.signal_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        
        # 신호 생성 설정
        self.min_confidence = config.get('min_confidence', 0.6)
        self.max_position_size = config.get('max_position_size', 0.1)
        self.stop_loss_multiplier = config.get('stop_loss_multiplier', 2.0)
        
        # 성과 모니터링
        self.total_signals = 0
        self.successful_signals = 0
        self.last_signal = None
        
        # 백테스팅 결과
        self.backtest_results = {}

    async def generate_trading_signal(
        self,
        news_analysis: Dict,
        impact_analysis: Dict,
        market_state: Dict,
        historical_performance: Dict
    ) -> Optional[TradingSignal]:
        """
        주어진 뉴스 분석 결과, 시장 영향 분석 결과, 
        현재 시장 상태, 과거 성과를 기반으로 트레이딩 신호를 생성한다.
        """
        try:
            # 1) 입력 데이터 유효성 검증
            if not self._validate_inputs(news_analysis, impact_analysis, market_state):
                return None

            # 2) 신호 강도 계산
            signal_strength = self._calculate_signal_strength(
                news_analysis,
                impact_analysis,
                historical_performance
            )

            # 3) 신뢰도 계산
            confidence = self._calculate_confidence(
                news_analysis,
                impact_analysis,
                historical_performance
            )

            # 신뢰도가 최소 기준치보다 낮으면 None
            if confidence < self.min_confidence:
                return None

            # 4) 포지션 크기 계산
            position_size = self._calculate_position_size(
                signal_strength,
                confidence,
                market_state
            )

            # 5) 목표가, 손절가 결정
            target_price, stop_loss = self._calculate_price_levels(
                market_state['current_price'],
                impact_analysis,
                signal_strength
            )

            # 6) 보유 기간(타임호라이즌) 결정
            time_horizon = self._determine_time_horizon(
                news_analysis,
                impact_analysis
            )

            # 7) 최종 신호 타입 결정
            signal_type = self._determine_signal_type(signal_strength)

            # 8) 트레이딩 신호 객체 생성
            signal = TradingSignal(
                timestamp=datetime.now(),
                signal_type=signal_type,
                confidence=confidence,
                position_size=position_size,
                target_price=target_price,
                stop_loss=stop_loss,
                time_horizon=time_horizon,
                signal_source='news_analysis',
                additional_info={
                    'news_sentiment': news_analysis['sentiment_score'],
                    'impact_score': impact_analysis.get('price_impact', 0),
                    'market_state': market_state['market_condition'],
                    'expected_return': signal_strength * confidence
                }
            )

            # 신호 기록
            self._record_signal(signal)
            return signal

        except Exception as e:
            logging.error(f"Signal generation error: {e}")
            return None

    def _validate_inputs(
        self,
        news_analysis: Dict,
        impact_analysis: Dict,
        market_state: Dict
    ) -> bool:
        """입력 데이터가 필수 필드를 모두 갖췄는지 검증"""
        required_news_fields = [
            'sentiment_score',
            'impact_score',
            'relevance_score'
        ]
        required_impact_fields = ['price_impact', 'volume_change']
        required_market_fields = [
            'current_price',
            'volatility',
            'market_condition'
        ]
        
        return all(field in news_analysis for field in required_news_fields) \
            and all(field in impact_analysis for field in required_impact_fields) \
            and all(field in market_state for field in required_market_fields)

    def _calculate_signal_strength(
        self,
        news_analysis: Dict,
        impact_analysis: Dict,
        historical_performance: Dict
    ) -> float:
        """뉴스 분석, 영향 분석, 과거 성과를 종합해 신호 강도를 -1~1 범위로 계산"""
        try:
            # 뉴스 분석 가중치
            news_weight = self._calculate_news_weight(news_analysis)
            # 영향 분석 가중치
            impact_weight = self._calculate_impact_weight(impact_analysis)
            # 과거 성과 가중치
            history_weight = self._calculate_history_weight(historical_performance)

            # 최종 신호 강도
            signal_strength = (
                news_weight * 0.4 +
                impact_weight * 0.4 +
                history_weight * 0.2
            )
            return np.clip(signal_strength, -1, 1)

        except Exception as e:
            logging.error(f"Signal strength calculation error: {e}")
            return 0.0

    def _calculate_news_weight(self, news_analysis: Dict) -> float:
        """뉴스 분석 결과(감성/영향/관련성)로부터 가중치 산출"""
        return (
            news_analysis['sentiment_score'] *
            news_analysis['impact_score'] *
            news_analysis['relevance_score']
        )

    def _calculate_impact_weight(self, impact_analysis: Dict) -> float:
        """시장 영향 분석 결과(가격 임팩트, 거래량 변화 등)로부터 가중치 산출"""
        price_impact = impact_analysis.get('price_impact', 0)
        volume_change = impact_analysis.get('volume_change', 0)
        # 가격 임팩트 × (1 + 가격 임팩트 부호에 따라 거래량 변동)
        return np.clip(
            price_impact * (1 + np.sign(price_impact) * volume_change),
            -1,
            1
        )

    def _calculate_history_weight(self, historical_performance: Dict) -> float:
        """
        과거 퍼포먼스(정확도 등)를 -1 ~ +1 범위로 매핑.
        예: accuracy=0.75면, 0.25 -> -0.5 = -1?, 이런 식으로 0.5를 기준으로 scale
        """
        accuracy = historical_performance.get('accuracy', 0.5)
        return (accuracy - 0.5) * 2  # 0.5 -> 0, 1.0 -> 1, 0.0 -> -1

    def _calculate_confidence(
        self,
        news_analysis: Dict,
        impact_analysis: Dict,
        historical_performance: Dict
    ) -> float:
        """신호 신뢰도(0~1) 계산"""
        sentiment_confidence = abs(news_analysis['sentiment_score'])
        impact_confidence = impact_analysis.get('significance', 0.5)
        historical_confidence = historical_performance.get('accuracy', 0.5)

        # 가중 평균
        confidence = (
            sentiment_confidence * 0.3 +
            impact_confidence * 0.4 +
            historical_confidence * 0.3
        )
        return np.clip(confidence, 0, 1)

    def _calculate_position_size(
        self,
        signal_strength: float,
        confidence: float,
        market_state: Dict
    ) -> float:
        """신호 강도와 신뢰도, 시장 변동성 등을 고려하여 포지션 크기 결정"""
        base_size = abs(signal_strength) * self.max_position_size
        confidence_adjusted = base_size * confidence

        # 시장 변동성이 높으면 포지션 축소
        volatility = market_state.get('volatility', 0)
        if volatility > 0.02:
            confidence_adjusted *= 0.8

        return min(confidence_adjusted, self.max_position_size)

    def _calculate_price_levels(
        self,
        current_price: float,
        impact_analysis: Dict,
        signal_strength: float
    ) -> Tuple[float, float]:
        """목표가, 손절가 계산"""
        expected_move = abs(signal_strength * impact_analysis.get('price_impact', 0))

        if signal_strength > 0:
            # 매수 신호
            target_price = current_price * (1 + expected_move)
            stop_loss = current_price * (1 - expected_move * self.stop_loss_multiplier)
        else:
            # 매도 신호
            target_price = current_price * (1 - expected_move)
            stop_loss = current_price * (1 + expected_move * self.stop_loss_multiplier)

        return target_price, stop_loss

    def _determine_time_horizon(
        self,
        news_analysis: Dict,
        impact_analysis: Dict
    ) -> int:
        """
        신호 보유 기간(초 단위) 결정.
        예: 단순히 1시간(3600초) 고정, 혹은 impact_analysis 내용에 따라 조정할 수 있음.
        """
        return 3600  # 예: 1시간

    def _determine_signal_type(self, signal_strength: float) -> str:
        """신호 강도 부호에 따라 buy / sell / hold"""
        if signal_strength > 0:
            return 'buy'
        elif signal_strength < 0:
            return 'sell'
        else:
            return 'hold'

    def _record_signal(self, signal: TradingSignal):
        """생성된 신호를 히스토리에 저장하고 카운트를 증가"""
        self.signal_history.append(signal)
        self.total_signals += 1
        self.last_signal = datetime.now()

    def get_performance_stats(self) -> Dict:
        """최근 100개 신호 기준 성과 통계"""
        if not self.signal_history:
            return {}
        recent_signals = list(self.signal_history)[-100:]
        
        buy_count = sum(1 for s in recent_signals if s.signal_type == 'buy')
        sell_count = sum(1 for s in recent_signals if s.signal_type == 'sell')
        hold_count = sum(1 for s in recent_signals if s.signal_type == 'hold')

        return {
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'accuracy': self.successful_signals / max(1, self.total_signals),
            'average_confidence': np.mean([s.confidence for s in recent_signals]),
            'average_position_size': np.mean([s.position_size for s in recent_signals]),
            'signal_distribution': {
                'buy': buy_count,
                'sell': sell_count,
                'hold': hold_count
            }
        }


class NewsBasedTradingManager:
    """뉴스 기반 트레이딩 통합 매니저"""
    def __init__(self, config: Dict):
        # 지연 로딩(import) - 순환 참조 방지
        from src.analyzers.news_analyzer import EnhancedNewsAnalyzer
        from src.analyzers.news_impact_analyzer import NewsImpactAnalyzer
        
        self.config = config
        self.news_analyzer = EnhancedNewsAnalyzer(config)
        self.impact_analyzer = NewsImpactAnalyzer(config)
        self.signal_generator = NewsSignalGenerator(config)
        
        # 실행 상태
        self.running = False
        self.last_update = None
        self.current_positions = []
        
        # 성능 모니터링
        self.execution_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit_loss': 0.0
        }

        # 백그라운드 태스크를 담아둘 리스트
        self.tasks = []

    async def start(self):
        """
        트레이딩 시스템 시작
        - 여기서는 create_task()로 무한 루프 태스크들을 만들고,
          즉시 반환하여 호출 측이 이어서 코드를 진행할 수 있게 함.
        """
        self.running = True
        logging.info("News-based trading system started")
        
        try:
            # 3개의 무한 루프 태스크를 백그라운드로 실행
            self.tasks = [
                asyncio.create_task(self._news_monitoring_loop()),
                asyncio.create_task(self._signal_processing_loop()),
                asyncio.create_task(self._position_monitoring_loop())
            ]
            # gather(...) 를 여기서 하지 않음.
            # -> test_integration() 등에서 함수가 즉시 반환되어
            #    3초 기다린 뒤 stop()을 호출할 수 있게 됨
        except Exception as e:
            logging.error(f"Trading system error: {e}")
            await self.stop()
            
    async def stop(self):
        """트레이딩 시스템 종료"""
        self.running = False
        logging.info("Stopping trading system...")
        
        # 열린 포지션 모두 정리 (예: 포지션 청산)
        await self._close_all_positions()

        # 백그라운드 태스크 취소 후 대기
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

        logging.info("Trading system stopped")

    async def _news_monitoring_loop(self):
        """뉴스 모니터링 루프 (무한 루프)"""
        while self.running:
            try:
                news_items = await self._fetch_news()
                if not news_items:
                    news_items = []

                for news in news_items:
                    # 뉴스 분석 (async)
                    analysis = await self.news_analyzer.analyze_news_with_market_impact(
                        news,
                        self._get_market_state()
                    )
                    
                    if analysis:
                        await self._process_news_analysis(news, analysis)
                        
                await asyncio.sleep(0.1)  # 100ms 주기
                
            except Exception as e:
                logging.error(f"News monitoring error: {e}")
                await asyncio.sleep(1)

    async def _signal_processing_loop(self):
        """신호 처리 루프 (무한 루프)"""
        while self.running:
            try:
                signals = await self._get_pending_signals()
                if not signals:
                    signals = []
                
                for signal in signals:
                    if await self._validate_signal(signal):
                        await self._execute_signal(signal)
                        
                await asyncio.sleep(0.01)  # 10ms 주기
                
            except Exception as e:
                logging.error(f"Signal processing error: {e}")
                await asyncio.sleep(0.1)

    async def _position_monitoring_loop(self):
        """포지션 모니터링 루프 (무한 루프)"""
        while self.running:
            try:
                for position in self.current_positions:
                    await self._check_position_status(position)
                await asyncio.sleep(0.01)
            except Exception as e:
                logging.error(f"Position monitoring error: {e}")
                await asyncio.sleep(0.1)

    async def _process_news_analysis(self, news, analysis):
        """
        뉴스 분석(NewsAnalysisResult) 결과를 받고,
        추가로 임팩트 분석 + 신호 생성 등 후속 작업을 수행.
        """
        try:
            # news는 NewsItem 객체, analysis는 NewsAnalysisResult 객체
            impact = await self.impact_analyzer.analyze_price_impact(
                news.timestamp,
                self._get_price_data(),
                self._get_volume_data()
            )

            # NewsAnalysisResult → dict 형태로 변환
            news_analysis_dict = {
                'sentiment_score': analysis.sentiment_score,
                'impact_score': analysis.impact_score,
                'relevance_score': analysis.relevance_score
            }

            # 여기서는 테스트용으로 임의 값을 준 예
            impact_dict = {
                'price_impact': 0.05,
                'volume_change': 0.1,
                'significance': 0.95
            }

            signal = await self.signal_generator.generate_trading_signal(
                news_analysis_dict,
                impact_dict,
                self._get_market_state(),
                self._get_historical_performance()
            )
            
            if signal:
                await self._queue_signal(signal)
                
        except Exception as e:
            logging.error(f"News analysis processing error: {e}")

    async def _execute_signal(self, signal: TradingSignal):
        """트레이딩 신호를 실제 매매로 실행"""
        try:
            order_result = await self._place_order(
                signal.signal_type,
                signal.position_size,
                signal.target_price,
                signal.stop_loss
            )
            
            if order_result['success']:
                # 포지션 목록에 추가
                self.current_positions.append({
                    'signal': signal,
                    'order': order_result,
                    'entry_time': datetime.now(),
                    'status': 'open'
                })
                self.execution_stats['total_trades'] += 1
            else:
                logging.error(f"Order execution failed: {order_result['error']}")
                self.execution_stats['failed_trades'] += 1
                
        except Exception as e:
            logging.error(f"Signal execution error: {e}")
            self.execution_stats['failed_trades'] += 1

    def get_system_status(self) -> Dict:
        """
        현재 시스템 상태(러닝 여부, 포지션 수, 분석기 스탯, 
        신호 생성기 스탯 등)를 반환
        """
        return {
            'running': self.running,
            'last_update': self.last_update,
            'current_positions': len(self.current_positions),
            'execution_stats': self.execution_stats,
            'news_analyzer_stats': self.news_analyzer.get_analysis_stats(),
            'impact_analyzer_stats': self.impact_analyzer.get_impact_stats(),
            'signal_generator_stats': self.signal_generator.get_performance_stats()
        }

    # -----------------------------------------------------------------------------
    # 아래 메서드들은 실제 구현 시 필요에 맞게 작성해야 하며,
    # 여기서는 테스트에서 "NoneType" 에러가 안 나게 하기 위해 최소한으로 구현.
    # -----------------------------------------------------------------------------

    async def _fetch_news(self) -> List:
        """새로운 뉴스 데이터 수집. 테스트용으로는 빈 리스트만 반환"""
        return []

    def _get_market_state(self) -> Dict:
        """현재 시장 상태 조회. 테스트용으로 간단히 반환"""
        return {
            'current_price': 50000000,
            'volatility': 0.02,
            'market_condition': 'normal'
        }

    def _get_price_data(self) -> List:
        """가격 데이터 조회. 테스트용으로 빈 리스트만 반환"""
        return []

    def _get_volume_data(self) -> List:
        """거래량 데이터 조회. 테스트용으로 빈 리스트만 반환"""
        return []

    async def _get_pending_signals(self) -> List:
        """대기 중인 신호 큐에서 신호 가져오기. 테스트용으론 빈 리스트"""
        return []

    async def _validate_signal(self, signal: TradingSignal) -> bool:
        """신호 유효성 검증. 테스트용으로 항상 True"""
        return True

    async def _place_order(
        self,
        order_type: str,
        size: float,
        target_price: float,
        stop_loss: float
    ) -> Dict:
        """주문 실행. 테스트용으로 항상 success=True"""
        return {"success": True}

    async def _check_position_status(self, position: Dict):
        """포지션 상태 확인/청산 등. 테스트용으로 아무 것도 하지 않음"""
        pass

    def _get_historical_performance(self) -> Dict:
        """과거 퍼포먼스 조회. 테스트용으로 임의 값 반환"""
        return {
            'accuracy': 0.75,
            'total_trades': 100,
            'success_rate': 0.8
        }

    async def _queue_signal(self, signal: TradingSignal):
        """생성된 신호를 대기열에 넣는 로직. 여기선 테스트용으로 아무 것도 안 함."""
        pass

    async def _close_all_positions(self):
        """종료 시 열린 포지션 모두 청산. 테스트용으로 리스트만 비움."""
        self.current_positions.clear()
