import asyncio
import aiohttp
import logging
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
import backoff
from textblob import TextBlob

class RateLimiter:
    """API 요청 제한 관리"""
    def __init__(self, max_requests: int, per_seconds: int):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.requests = deque()

    async def acquire(self):
        now = time.time()
        while self.requests and now - self.requests[0] > self.per_seconds:
            self.requests.popleft()

        if len(self.requests) >= self.max_requests:
            sleep_time = self.requests[0] + self.per_seconds - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.requests.append(now)

@dataclass
class RedditContent:
    id: str
    timestamp: datetime
    title: str
    content: str
    source: str
    subreddit: str
    score: int
    upvote_ratio: float
    comment_count: int
    author: str
    is_self: bool
    url: str
    created_utc: float
    sentiment_score: float  # ✅ 감성 분석 점수 추가

    def analyze_sentiment(self):
        """ 감성 분석 수행 """
        text = self.title + " " + self.content
        self.sentiment_score = TextBlob(text).sentiment.polarity

class RedditCollector:
    """개선된 레딧 데이터 수집기"""
    def __init__(self, config: Dict):
        self.client_id = config['reddit_client_id']
        self.client_secret = config['reddit_client_secret']
        self.user_agent = f"python:crypto.analyzer:v1.0 (by /u/{config['reddit_username']})"
        self.subreddits = config.get('subreddits', ['CryptoCurrency', 'Bitcoin', 'ethereum'])
        self.rate_limiter = RateLimiter(max_requests=60, per_seconds=60)
        self.access_token = None
        self.token_expires_at = None
        self.session = None
        self.db = sqlite3.connect("reddit_data.db")
        self._setup_db()

    def _setup_db(self):
        """ 데이터베이스 초기화 """
        query = """
        CREATE TABLE IF NOT EXISTS reddit_posts (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            title TEXT,
            content TEXT,
            subreddit TEXT,
            score INTEGER,
            upvote_ratio REAL,
            comment_count INTEGER,
            author TEXT,
            is_self BOOLEAN,
            url TEXT,
            sentiment_score REAL
        )
        """
        self.db.execute(query)
        self.db.commit()

    async def initialize(self):
        """수집기 초기화"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        await self._refresh_token()

    async def cleanup(self):
        """리소스 정리"""
        if self.session:
            await self.session.close()
            self.session = None
        self.db.close()

    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=5)
    async def _fetch_subreddit_posts(self, subreddit: str, limit: int = 100) -> List[RedditContent]:
        """서브레딧 게시물 수집"""
        url = f"https://oauth.reddit.com/r/{subreddit}/hot?limit={limit}"
        headers = {"Authorization": f"Bearer {self.access_token}", "User-Agent": self.user_agent}

        await self.rate_limiter.acquire()

        async with self.session.get(url, headers=headers) as response:
            if response.status == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                await asyncio.sleep(retry_after)
                return []

            response.raise_for_status()
            data = await response.json()

            posts = []
            for post in data["data"]["children"]:
                post_data = post["data"]
                content = RedditContent(
                    id=post_data["id"],
                    timestamp=datetime.fromtimestamp(post_data["created_utc"]),
                    title=post_data["title"],
                    content=post_data.get("selftext", ""),
                    source=f"reddit/r/{subreddit}",
                    subreddit=post_data["subreddit"],
                    score=post_data["score"],
                    upvote_ratio=post_data["upvote_ratio"],
                    comment_count=post_data["num_comments"],
                    author=post_data["author"],
                    is_self=post_data["is_self"],
                    url=post_data["url"],
                    created_utc=post_data["created_utc"],
                    sentiment_score=0.0
                )
                content.analyze_sentiment()
                posts.append(content)

            return posts

    async def collect(self):
        """병렬 데이터 수집"""
        tasks = [self._fetch_subreddit_posts(subreddit) for subreddit in self.subreddits]
        results = await asyncio.gather(*tasks)

        all_posts = [post for sublist in results for post in sublist]

        for post in all_posts:
            self._save_to_db(post)

    def _save_to_db(self, post: RedditContent):
        """ 데이터베이스에 저장 (중복 방지) """
        query = """
        INSERT OR IGNORE INTO reddit_posts
        (id, timestamp, title, content, subreddit, score, upvote_ratio, comment_count, author, is_self, url, sentiment_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute(query, (post.id, post.timestamp, post.title, post.content, post.subreddit,
                                post.score, post.upvote_ratio, post.comment_count, post.author,
                                post.is_self, post.url, post.sentiment_score))
        self.db.commit()
