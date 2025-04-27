# src/collectors/base_collector.py

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque  # 이 줄 추가

@dataclass
class BaseContent:
    """기본 콘텐츠 데이터 클래스"""
    id: str
    timestamp: datetime
    title: str
    content: str
    source: str
    language: str = 'en'  # 기본값이 있는 필드는 마지막으로

    def validate(self) -> bool:
        """데이터 유효성 검증"""
        return all([
            isinstance(self.id, str) and self.id.strip(),
            isinstance(self.timestamp, datetime),
            isinstance(self.title, str) and self.title.strip(),
            isinstance(self.content, str),
            isinstance(self.source, str) and self.source.strip(),
            isinstance(self.language, str)
        ])

class BaseCollector(ABC):
    """기본 수집기 추상 클래스"""
    def __init__(self, config: Dict):
        # 기본 설정
        self.config = config
        self.cache = deque(maxlen=config.get('cache_size', 1000))
        self.last_update = None
        self.running = False
        
        # 성능 모니터링
        self.collected_count = 0
        self.error_count = 0
        self.start_time = datetime.now()

        # 로깅 설정
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_logging()

    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=self.config.get('log_level', logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.config.get('log_file'),
            filemode='a'
        )

    @abstractmethod
    async def collect(self) -> List[BaseContent]:
        """데이터 수집 추상 메서드"""
        pass

    async def start_collection(self):
        """수집 시작"""
        self.running = True
        self.logger.info(f"Starting {self.__class__.__name__} collection")
        
        try:
            while self.running:
                try:
                    # 데이터 수집 실행
                    items = await self.collect()
                    
                    # 유효성 검증
                    valid_items = [
                        item for item in items
                        if item.validate()
                    ]
                    
                    # 캐시에 저장
                    for item in valid_items:
                        self.cache.append(item)
                        self.collected_count += 1
                    
                    self.last_update = datetime.now()
                    
                    # 통계 로깅
                    if self.collected_count % 100 == 0:  # 100개마다 로깅
                        self.logger.info(self.get_stats_summary())
                    
                    # 수집 간격 대기
                    await asyncio.sleep(
                        self.config.get('update_interval', 60)
                    )
                    
                except Exception as e:
                    self.error_count += 1
                    self.logger.error(f"Collection error: {str(e)}", exc_info=True)
                    await asyncio.sleep(5)  # 오류 시 더 긴 대기
                    
        except asyncio.CancelledError:
            self.logger.info(f"{self.__class__.__name__} collection stopped")
            self.running = False
            await self.cleanup()

    async def stop_collection(self):
        """수집 중지"""
        self.running = False
        self.logger.info(f"Stopping {self.__class__.__name__} collection...")
        await self.cleanup()

    async def cleanup(self):
        """리소스 정리"""
        pass  # 하위 클래스에서 필요한 정리 작업 구현

    def get_latest_content(self, limit: int = 10) -> List[BaseContent]:
        """최신 콘텐츠 조회"""
        return list(self.cache)[-limit:]

    def get_stats(self) -> Dict:
        """수집 통계 정보"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            'collector_type': self.__class__.__name__,
            'collected_count': self.collected_count,
            'error_count': self.error_count,
            'cache_size': len(self.cache),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'uptime_seconds': uptime,
            'collection_rate': self.collected_count / max(1, uptime) * 3600,  # 시간당
            'error_rate': self.error_count / max(1, uptime) * 3600,  # 시간당
            'success_rate': (
                (self.collected_count - self.error_count) /
                max(1, self.collected_count) * 100
            )
        }

    def get_stats_summary(self) -> str:
        """통계 요약 문자열"""
        stats = self.get_stats()
        return (
            f"{stats['collector_type']} Stats:\n"
            f"Collected: {stats['collected_count']}, "
            f"Errors: {stats['error_count']}, "
            f"Success Rate: {stats['success_rate']:.1f}%, "
            f"Collection Rate: {stats['collection_rate']:.1f}/hour"
        )