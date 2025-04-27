# src/news_collector.py

import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import feedparser
import time
from bs4 import BeautifulSoup

from ..analyzers.news_analyzer import NewsItem

@dataclass
class NewsSource:
    """뉴스 소스 설정"""
    name: str
    url: str
    update_interval: float  # seconds
    source_type: str  # 'rss', 'api', 'html'
    category: Optional[str] = None  # 특정 카테고리 URL이나 파라미터
    api_key: Optional[str] = None
    language: str = 'en'  # 기본값을 영어로 변경

class NewsCollector:
    """실시간 뉴스 수집기"""
    def __init__(self, config: Dict):
        self.config = config
        self.sources = self._init_sources()
        self.news_cache = deque(maxlen=1000)
        self.last_update = None
        self.running = False
        
        # 성능 모니터링
        self.collected_count = 0
        self.error_count = 0
        
        # 스팸 필터링 (영문 키워드 추가)
        self.spam_keywords = [
            'sponsored', 'advertisement', 'promoted',
            'partner content', 'sponsored content',
            'press release', 'promotional feature'
        ]
        
        # 암호화폐 관련 키워드
        self.crypto_keywords = [
            'bitcoin', 'crypto', 'cryptocurrency', 'blockchain',
            'ethereum', 'digital currency', 'digital asset',
            'crypto market', 'defi', 'nft', 'web3'
        ]

    def _init_sources(self) -> List[NewsSource]:
        """뉴스 소스 초기화"""
        return [
            # BBC News
            NewsSource(
                name="BBC News",
                url="https://feeds.bbci.co.uk/news/technology/rss.xml",
                update_interval=60,   # 1분
                source_type="rss"
            ),
            NewsSource(
                name="BBC Business",
                url="https://feeds.bbci.co.uk/news/business/rss.xml",
                update_interval=300,
                source_type="rss"
            ),
            
            # CNN
            NewsSource(
                name="CNN Business",
                url="http://rss.cnn.com/rss/edition_business.rss",
                update_interval=300,
                source_type="rss"
            ),
            NewsSource(
                name="CNN Technology",
                url="http://rss.cnn.com/rss/edition_technology.rss",
                update_interval=300,
                source_type="rss"
            ),
            
            # Fox News
            NewsSource(
                name="Fox Business",
                url="https://moxie.foxbusiness.com/feedburner/feedburner.rss",
                update_interval=300,
                source_type="rss"
            ),
            NewsSource(
                name="Fox Technology",
                url="https://moxie.foxnews.com/feedburner/scitech.rss",
                update_interval=300,
                source_type="rss"
            )
        ]

    async def start_collection(self):
        """뉴스 수집 시작"""
        self.running = True
        collection_tasks = []
        
        try:
            for source in self.sources:
                task = asyncio.create_task(
                    self._collect_from_source(source)
                )
                collection_tasks.append(task)
            
            await asyncio.gather(*collection_tasks)
            
        except Exception as e:
            logging.error(f"News collection error: {e}")
            self.running = False

    async def _collect_from_source(self, source: NewsSource):
        """특정 소스에서 뉴스 수집"""
        async with aiohttp.ClientSession() as session:
            while self.running:
                try:
                    news_items = await self._collect_from_rss(source)
                    
                    # 암호화폐 관련 뉴스만 필터링
                    crypto_news = [
                        item for item in news_items
                        if self._is_crypto_related(item) and not self._is_spam(item)
                    ]
                    
                    # 캐시에 저장
                    for item in crypto_news:
                        self.news_cache.append(item)
                        self.collected_count += 1
                    
                    self.last_update = datetime.now()
                    await asyncio.sleep(source.update_interval)
                    
                except Exception as e:
                    self.error_count += 1
                    logging.error(f"Source collection error: {e}")
                    await asyncio.sleep(5)

    async def _collect_from_rss(self, source: NewsSource) -> List[NewsItem]:
        """RSS 피드에서 뉴스 수집"""
        try:
            feed = feedparser.parse(source.url)
            news_items = []

            for entry in feed.entries:
                # published_parsed가 없는 경우 현재 시간 사용
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    timestamp = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                else:
                    timestamp = datetime.now()

                # description이 없는 경우 빈 문자열 사용
                content = getattr(entry, 'description', '')
                if not content:
                    content = getattr(entry, 'summary', '')

                news_item = NewsItem(
                    timestamp=timestamp,
                    title=entry.title.strip(),
                    content=content.strip(),
                    source=source.name,
                    language=source.language
                )
                
                if news_item.validate():
                    news_items.append(news_item)

            return news_items

        except Exception as e:
            logging.error(f"RSS collection error from {source.name}: {e}")
            return []

    def _is_crypto_related(self, news_item: NewsItem) -> bool:
        """암호화폐 관련 뉴스인지 확인"""
        text = (news_item.title + ' ' + news_item.content).lower()
        return any(keyword in text for keyword in self.crypto_keywords)

    def _is_spam(self, news_item: NewsItem) -> bool:
        """스팸 뉴스 필터링"""
        text = (news_item.title + ' ' + news_item.content).lower()
        
        # 스팸 키워드 체크
        if any(keyword in text for keyword in self.spam_keywords):
            return True
            
        # 제목 중복 체크
        if any(item.title == news_item.title for item in self.news_cache):
            return True
            
        # 최소 길이 체크
        if len(news_item.content) < 100:
            return True
            
        return False

    def get_latest_news(self, limit: int = 10) -> List[NewsItem]:
        """최신 뉴스 조회"""
        return list(self.news_cache)[-limit:]

    def get_collection_stats(self) -> Dict:
        """수집 통계 정보"""
        return {
            'collected_count': self.collected_count,
            'error_count': self.error_count,
            'cache_size': len(self.news_cache),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'sources': [source.name for source in self.sources],
            'crypto_news_ratio': self._calculate_crypto_news_ratio()
        }
    
    def _calculate_crypto_news_ratio(self) -> float:
        """암호화폐 관련 뉴스 비율 계산"""
        if not self.news_cache:
            return 0.0
        
        crypto_news_count = sum(
            1 for item in self.news_cache
            if self._is_crypto_related(item)
        )
        
        return crypto_news_count / len(self.news_cache)