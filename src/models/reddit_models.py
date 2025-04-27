from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import numpy as np

@dataclass
class RedditPost:
    """레딧 게시물 데이터 모델"""
    id: str
    created_at: datetime
    title: str
    content: str
    subreddit: str
    author: str
    upvotes: int
    downvotes: int
    score: float
    comment_count: int
    sentiment_score: float
    is_daily_thread: bool = False

    def calculate_popularity(self) -> float:
        """게시물 인기도 점수 계산"""
        base_score = self.upvotes / max(1, (self.upvotes + self.downvotes))
        comment_weight = np.log1p(self.comment_count)
        return base_score * (1 + 0.5 * comment_weight)

@dataclass
class RedditComment:
    """레딧 댓글 데이터 모델"""
    id: str
    post_id: str
    created_at: datetime
    content: str
    author: str
    score: int
    sentiment_score: float
    parent_id: Optional[str] = None

@dataclass
class RedditMetrics:
    """레딧 종합 메트릭스"""
    timestamp: datetime
    total_posts: int
    total_comments: int
    average_sentiment: float
    bullish_ratio: float  # 강세 게시물 비율
    trending_keywords: List[str]
    popularity_score: float
    volatility_index: float  # 논의 활성도

    @property
    def market_signal(self) -> str:
        """시장 시그널 판단"""
        if self.bullish_ratio > 0.7 and self.popularity_score > 0.8:
            return "very_bullish"
        elif self.bullish_ratio > 0.6:
            return "bullish"
        elif self.bullish_ratio < 0.3:
            return "bearish"
        elif self.bullish_ratio < 0.2 and self.popularity_score > 0.8:
            return "very_bearish"
        return "neutral"

    @property
    def risk_level(self) -> str:
        """리스크 레벨 평가"""
        if self.volatility_index > 0.8:
            return "high"
        elif self.volatility_index > 0.5:
            return "medium"
        return "low"

@dataclass
class SubredditActivity:
    """서브레딧 활동 지표"""
    subreddit: str
    timestamp: datetime
    post_count: int
    comment_count: int
    unique_authors: int
    average_score: float
    sentiment_trend: float
    hot_topics: List[str]

    def get_activity_level(self) -> str:
        """활동 수준 평가"""
        if self.post_count > 100 and self.comment_count > 1000:
            return "very_high"
        elif self.post_count > 50 and self.comment_count > 500:
            return "high"
        elif self.post_count > 20 and self.comment_count > 200:
            return "medium"
        return "low"