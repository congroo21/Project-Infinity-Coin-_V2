import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np

from src.models.openai_client import OpenAIClient

@dataclass
class NewsItem:
    """뉴스 데이터 클래스"""
    timestamp: datetime
    title: str
    content: str
    source: str
    language: str = 'ko'
    
    def validate(self) -> bool:
        """데이터 유효성 검증"""
        return all([
            isinstance(self.timestamp, datetime),
            isinstance(self.title, str) and self.title.strip(),
            isinstance(self.content, str) and self.content.strip(),
            isinstance(self.source, str) and self.source.strip()
        ])

@dataclass
class NewsAnalysisResult:
    """향상된 뉴스 분석 결과"""
    news_item: NewsItem
    sentiment_score: float      # -1 (매우 부정) ~ 1 (매우 긍정)
    impact_score: float         # 0 (영향 없음) ~ 1 (매우 큰 영향)
    relevance_score: float      # 0 (관련 없음) ~ 1 (매우 관련)
    keywords: List[str]
    summary: str
    timestamp: datetime
    price_impacts: Optional[Dict[str, float]] = None  # 각 시간프레임별 가격 영향
    accuracy_metrics: Optional[Dict[str, float]] = None  # 정확도 메트릭스
    alpha_signal: Optional[Dict[str, float]] = None  # 알파 시그널

@dataclass
class MarketState:
    """시장 상태 데이터"""
    current_price: float
    volatility: float
    trend: float
    volume: float
    timestamp: datetime
    market_condition: str = 'normal'

class EnhancedNewsAnalyzer:
    """향상된 뉴스 분석 시스템"""
    def __init__(self, config: Dict = None):
        self.config = config or {}  # config가 None이면 빈 딕셔너리 사용
        self.openai_client = OpenAIClient()
        self.news_cache = deque(maxlen=1000)
        self.analysis_cache = deque(maxlen=1000)
        self.price_history = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=1000)
        self.last_analysis = None
        
        # 성능 모니터링
        self.processed_count = 0
        self.error_count = 0
        
        # 분석 시간프레임 설정
        self.timeframes = [1, 5, 10, 30, 60]  # 초 단위
        
        # GPT-4 프롬프트 템플릿
        self.analysis_prompt_template = """
        다음 뉴스를 분석하여 암호화폐 시장에 미치는 영향을 평가해주세요:

        제목: {title}
        내용: {content}
        출처: {source}
        시장상황: {market_context}

        다음 형식으로 응답해주세요:
        1. 감성 점수 (-1 ~ 1):
        2. 영향도 점수 (0 ~ 1):
        3. 관련성 점수 (0 ~ 1):
        4. 주요 키워드 (쉼표로 구분):
        5. 요약 (한 문장):
        6. 예상 가격 영향 시간:
        7. 거래 제안:
        """

    def _check_cache(self, news_item: NewsItem) -> Optional[NewsAnalysisResult]:
        """캐시된 분석 결과 확인"""
        try:
            for result in self.analysis_cache:
                if (result.news_item.title == news_item.title and
                    result.news_item.source == news_item.source):
                    return result
            return None
        except Exception as e:
            logging.error(f"Cache check error: {e}")
            return None

    async def analyze_news_with_market_impact(
        self,
        news_item: NewsItem,
        market_state: MarketState
    ) -> Optional[NewsAnalysisResult]:
        """향상된 뉴스 분석 실행 (비동기 함수이지만, openai_client는 동기로 호출)"""
        try:
            if not news_item.validate():
                raise ValueError("Invalid news data")

            # 캐시 확인
            cached_result = self._check_cache(news_item)
            if cached_result:
                logging.info("캐시된 분석 결과 사용")
                return cached_result

            # GPT-4 분석 요청 (동기 호출 -> await 제거)
            market_context = self._get_market_context(market_state)
            prompt = self.analysis_prompt_template.format(
                title=news_item.title,
                content=news_item.content,
                source=news_item.source,
                market_context=market_context
            )
            response = self.openai_client.generate_response(prompt)

            # 응답 파싱
            analysis_dict = self._parse_gpt_response(response)
            if not analysis_dict:
                raise ValueError("Failed to parse GPT response")

            # 가격 영향 분석 (여기서는 async)
            price_impacts = await self._analyze_price_impact(
                news_item,
                list(self.price_history),
                market_state
            )

            # 정확도 검증
            accuracy_metrics = self._validate_sentiment_accuracy(
                analysis_dict['sentiment_score'],
                price_impacts
            )

            # 알파 시그널 생성
            alpha_signal = self._generate_alpha_signal(
                analysis_dict,
                market_state,
                accuracy_metrics
            )

            # 결과 생성
            result = NewsAnalysisResult(
                news_item=news_item,
                sentiment_score=analysis_dict['sentiment_score'],
                impact_score=analysis_dict['impact_score'],
                relevance_score=analysis_dict['relevance_score'],
                keywords=analysis_dict['keywords'],
                summary=analysis_dict['summary'],
                timestamp=datetime.now(),
                price_impacts=price_impacts,
                accuracy_metrics=accuracy_metrics,
                alpha_signal=alpha_signal
            )

            # 캐시에 저장
            self.analysis_cache.append(result)
            self.accuracy_history.append(accuracy_metrics)
            self.processed_count += 1
            self.last_analysis = datetime.now()

            return result

        except Exception as e:
            self.error_count += 1
            logging.error(f"News analysis error: {e}")
            return None

    def _parse_gpt_response(self, response: str) -> Optional[Dict]:
        """GPT 응답 파싱"""
        try:
            lines = response.strip().split('\n')
            result = {}
            
            for line in lines:
                line = line.strip()
                if '감성 점수' in line:
                    val = line.split(':')[1].strip()
                    result['sentiment_score'] = float(val)
                elif '영향도 점수' in line:
                    val = line.split(':')[1].strip()
                    result['impact_score'] = float(val)
                elif '관련성 점수' in line:
                    val = line.split(':')[1].strip()
                    result['relevance_score'] = float(val)
                elif '주요 키워드' in line:
                    keywords = line.split(':')[1].strip()
                    result['keywords'] = [k.strip() for k in keywords.split(',')]
                elif '요약' in line:
                    val = line.split(':')[1].strip()
                    result['summary'] = val

            required_fields = [
                'sentiment_score', 'impact_score', 'relevance_score',
                'keywords', 'summary'
            ]
            if all(field in result for field in required_fields):
                return result
            return None

        except Exception as e:
            logging.error(f"Response parsing error: {e}")
            return None

    def _get_market_context(self, market_state: MarketState) -> str:
        """시장 상황 문자열 생성"""
        return (
            f"현재가: {market_state.current_price:,.0f}, "
            f"변동성: {market_state.volatility:.4f}, "
            f"추세: {market_state.trend:+.4f}, "
            f"거래량: {market_state.volume:.1f}"
        )

    async def _analyze_price_impact(
        self,
        news_item: NewsItem,
        price_history: List[float],
        market_state: MarketState
    ) -> Dict[str, float]:
        """가격 영향 분석 (비동기)"""
        impacts = {}
        try:
            for tf in self.timeframes:
                tf_change = self._estimate_price_change(
                    market_state.current_price,
                    market_state.volatility,
                    market_state.trend,
                    tf
                )
                impacts[f"{tf}s"] = tf_change
            return impacts
        except Exception as e:
            logging.error(f"Price impact analysis error: {e}")
            return {}

    def _estimate_price_change(
        self,
        current_price: float,
        volatility: float,
        trend: float,
        timeframe: int
    ) -> float:
        """시간프레임별 가격 변화율 추정"""
        t = timeframe / (24 * 3600)
        drift = trend * t
        random_component = volatility * np.sqrt(t) * np.random.normal()
        return drift + random_component

    # src/analyzers/news_analyzer.py의 리스크 관련 함수 수정

    def _validate_sentiment_accuracy(
        self,
        sentiment_score: float,
        price_impacts: Dict[str, float]
    ) -> Dict[str, float]:
        """감성 분석 정확도 검증"""
        try:
            # 중앙화된 리스크 계산기 사용
            from src.utils.risk_utils import risk_calculator
            
            results = {}
            for timeframe, price_change in price_impacts.items():
                sentiment_direction = np.sign(sentiment_score)
                price_direction = np.sign(price_change)
                direction_match = sentiment_direction == price_direction
                magnitude_error = abs(abs(sentiment_score) - abs(price_change))

                results[f"{timeframe}_direction_match"] = float(direction_match)
                results[f"{timeframe}_magnitude_error"] = magnitude_error

            # 방향 일치 정확도 계산
            direction_matches = [
                v for k, v in results.items() if 'direction_match' in k
            ]
            overall_accuracy = np.mean(direction_matches) if direction_matches else 0.5
            results['overall_accuracy'] = overall_accuracy
            
            return results
            
        except Exception as e:
            logging.error(f"Accuracy validation error: {e}")
            return {'overall_accuracy': 0.5}

    def _generate_alpha_signal(
        self,
        analysis_result: Dict,
        market_state: MarketState,
        accuracy_metrics: Dict
    ) -> Dict:
        """알파 시그널 생성"""
        try:
            # 중요도 점수 계산
            importance_score = (
                analysis_result['sentiment_score'] * 0.3 +
                analysis_result['impact_score'] * 0.4 +
                analysis_result['relevance_score'] * 0.3
            )

            # 중앙화된 리스크 계산기 사용
            from src.utils.risk_utils import risk_calculator
            
            # 시장 조정 계수 계산
            market_adjustment = 1.0
            if market_state.volatility > 0.02:  # 고변동성
                market_adjustment *= 0.8
            if abs(market_state.trend) > 0.01:  # 강한 추세
                market_adjustment *= 1.2

            # 정확도 조정
            accuracy_adjustment = accuracy_metrics.get('overall_accuracy', 0.5)

            # 최종 시그널 계산
            signal_strength = importance_score * market_adjustment * accuracy_adjustment
            signal_direction = np.sign(analysis_result['sentiment_score'])

            return {
                'signal': signal_strength * signal_direction,
                'confidence': importance_score * accuracy_adjustment,
                'suggested_position_size': min(0.1, abs(signal_strength)),
                'hold_duration': int(30 * importance_score),
                'market_adjustment': market_adjustment,
                'accuracy_adjustment': accuracy_adjustment
            }
            
        except Exception as e:
            logging.error(f"Alpha signal generation error: {e}")
            return {}

    def get_analysis_stats(self) -> Dict:
        """분석 통계"""
        if not self.analysis_cache:
            return {}
        recent_results = list(self.analysis_cache)[-100:]
        accuracy_stats = list(self.accuracy_history)[-100:]

        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'average_sentiment': np.mean([r.sentiment_score for r in recent_results]),
            'average_impact': np.mean([r.impact_score for r in recent_results]),
            'average_accuracy': np.mean([a['overall_accuracy'] for a in accuracy_stats]),
            'latest_analysis': self.last_analysis.isoformat() if self.last_analysis else None
        }
