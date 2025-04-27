import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from textblob import TextBlob

from ..models.reddit_models import (
    RedditPost, RedditComment, RedditMetrics, SubredditActivity
)

class RedditAnalyzer:
    """레딧 데이터 분석기"""
    def __init__(self):
        self.sentiment_cache = {}
        self.keyword_weights = {
            "bull": 1.5, "bullish": 1.5, "moon": 1.2,
            "bear": -1.5, "bearish": -1.5, "crash": -1.2,
            "buy": 1.0, "sell": -1.0,
            "pump": 0.8, "dump": -0.8
        }
        self.min_word_length = 3
        
    def analyze_post(self, post: RedditPost) -> Dict:
        """개별 게시물 분석"""
        try:
            # 감성 분석
            if post.id not in self.sentiment_cache:
                combined_text = f"{post.title} {post.content}"
                sentiment = TextBlob(combined_text).sentiment.polarity
                self.sentiment_cache[post.id] = sentiment
            
            # 키워드 가중치 적용
            weighted_score = self._calculate_keyword_score(
                f"{post.title} {post.content}"
            )
            
            # 최종 점수 계산
            final_score = (
                self.sentiment_cache[post.id] * 0.5 +
                weighted_score * 0.3 +
                post.calculate_popularity() * 0.2
            )
            
            return {
                'sentiment': self.sentiment_cache[post.id],
                'weighted_score': weighted_score,
                'final_score': final_score,
                'is_bullish': final_score > 0.2
            }
            
        except Exception as e:
            logging.error(f"게시물 분석 오류: {e}")
            return {}

    def _calculate_keyword_score(self, text: str) -> float:
        """키워드 기반 점수 계산"""
        words = text.lower().split()
        total_score = 0
        matched_words = 0
        
        for word in words:
            if len(word) >= self.min_word_length:
                if word in self.keyword_weights:
                    total_score += self.keyword_weights[word]
                    matched_words += 1
                    
        return total_score / max(1, matched_words)

    def analyze_comments(
        self,
        comments: List[RedditComment]
    ) -> Dict:
        """댓글 분석"""
        try:
            if not comments:
                return {}
                
            sentiments = []
            total_score = 0
            
            for comment in comments:
                if comment.id not in self.sentiment_cache:
                    sentiment = TextBlob(comment.content).sentiment.polarity
                    self.sentiment_cache[comment.id] = sentiment
                
                sentiments.append(self.sentiment_cache[comment.id])
                total_score += comment.score
                
            return {
                'average_sentiment': np.mean(sentiments),
                'sentiment_std': np.std(sentiments),
                'total_score': total_score,
                'unique_authors': len(set(c.author for c in comments))
            }
            
        except Exception as e:
            logging.error(f"댓글 분석 오류: {e}")
            return {}

    def calculate_metrics(
        self,
        posts: List[RedditPost],
        comments: List[RedditComment]
    ) -> Optional[RedditMetrics]:
        """종합 메트릭스 계산"""
        try:
            if not posts:
                return None
                
            # 게시물 분석
            post_analyses = [
                self.analyze_post(post) for post in posts
            ]
            
            # 댓글 분석
            comment_analysis = self.analyze_comments(comments)
            
            # 인기 키워드 추출
            trending_keywords = self._extract_trending_keywords(posts)
            
            # 강세/약세 비율 계산
            bullish_count = sum(
                1 for analysis in post_analyses
                if analysis.get('is_bullish', False)
            )
            
            # 변동성 지수 계산
            sentiment_values = [
                analysis.get('sentiment', 0)
                for analysis in post_analyses
            ]
            volatility = np.std(sentiment_values) if sentiment_values else 0
            
            # 종합 인기도 점수
            popularity = np.mean([
                post.calculate_popularity()
                for post in posts
            ])

            return RedditMetrics(
                timestamp=datetime.now(),
                total_posts=len(posts),
                total_comments=len(comments),
                average_sentiment=np.mean([
                    analysis.get('sentiment', 0)
                    for analysis in post_analyses
                ]),
                bullish_ratio=bullish_count / len(posts),
                trending_keywords=trending_keywords,
                popularity_score=popularity,
                volatility_index=volatility
            )
            
        except Exception as e:
            logging.error(f"메트릭스 계산 오류: {e}")
            return None

    def _extract_trending_keywords(
        self,
        posts: List[RedditPost],
        min_count: int = 3
    ) -> List[str]:
        """트렌딩 키워드 추출"""
        try:
            word_counts = defaultdict(int)
            
            for post in posts:
                words = set(
                    word.lower()
                    for word in f"{post.title} {post.content}".split()
                    if len(word) >= self.min_word_length
                )
                
                for word in words:
                    word_counts[word] += 1
            
            # 최소 출현 횟수 이상인 키워드만 선택
            trending = [
                word for word, count in word_counts.items()
                if count >= min_count
            ]
            
            # 출현 빈도순 정렬
            return sorted(
                trending,
                key=lambda x: word_counts[x],
                reverse=True
            )[:10]  # 상위 10개만
            
        except Exception as e:
            logging.error(f"키워드 추출 오류: {e}")
            return []

    def calculate_subreddit_activity(
        self,
        subreddit: str,
        posts: List[RedditPost],
        comments: List[RedditComment]
    ) -> Optional[SubredditActivity]:
        """서브레딧 활동 분석"""
        try:
            if not posts:
                return None
                
            # 최근 24시간 데이터만 필터링
            recent_threshold = datetime.now() - timedelta(days=1)
            recent_posts = [
                post for post in posts
                if post.created_at >= recent_threshold
            ]
            recent_comments = [
                comment for comment in comments
                if comment.created_at >= recent_threshold
            ]
            
            # 감성 트렌드 계산
            hourly_sentiments = defaultdict(list)
            for post in recent_posts:
                hour = post.created_at.hour
                if post.id in self.sentiment_cache:
                    hourly_sentiments[hour].append(
                        self.sentiment_cache[post.id]
                    )
                    
            sentiment_trend = 0.0
            if hourly_sentiments:
                hourly_avg = [
                    np.mean(sentiments)
                    for sentiments in hourly_sentiments.values()
                ]
                sentiment_trend = np.polyfit(
                    range(len(hourly_avg)),
                    hourly_avg,
                    deg=1
                )[0]

            return SubredditActivity(
                subreddit=subreddit,
                timestamp=datetime.now(),
                post_count=len(recent_posts),
                comment_count=len(recent_comments),
                unique_authors=len(set(
                    post.author for post in recent_posts
                )),
                average_score=np.mean([
                    post.score for post in recent_posts
                ]),
                sentiment_trend=sentiment_trend,
                hot_topics=self._extract_trending_keywords(
                    recent_posts,
                    min_count=2
                )
            )
            
        except Exception as e:
            logging.error(f"서브레딧 활동 분석 오류: {e}")
            return None