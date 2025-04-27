# my_project/src/data_collector.py

import asyncio
import logging
import concurrent.futures
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass
from collections import deque

import pyupbit
from pyupbit import WebSocketManager

# DatabaseManager 임포트
from src.database import DatabaseManager


@dataclass
class CollectorConfig:
    """데이터 수집기 설정"""
    ticker: str = "KRW-BTC"
    interval: str = "minute1"
    limit: int = 200
    cache_size: int = 1000
    cache_duration_ms: int = 50
    retry_attempts: int = 3
    retry_delay: float = 0.1


@dataclass
class MarketData:
    """시장 데이터 클래스"""
    timestamp: datetime
    price: float
    volume: float
    orderbook: Optional[Dict] = None

    def validate(self) -> bool:
        """데이터 유효성 검증"""
        try:
            if not isinstance(self.timestamp, datetime):
                logging.error("Invalid timestamp type")
                return False

            if not isinstance(self.price, (int, float)) or self.price <= 0:
                logging.error(f"Invalid price: {self.price}")
                return False

            if not isinstance(self.volume, (int, float)) or self.volume < 0:
                logging.error(f"Invalid volume: {self.volume}")
                return False

            if self.orderbook:
                if not isinstance(self.orderbook, dict):
                    logging.error("Invalid orderbook type")
                    return False
                if 'orderbook_units' not in self.orderbook:
                    logging.error("Missing orderbook_units in orderbook data")
                    return False

            return True

        except Exception as e:
            logging.error(f"Data validation error: {e}")
            return False


class BaseCollector(ABC):
    """개선된 기본 수집기 클래스 (추상 클래스)"""
    def __init__(self, config: CollectorConfig, db_manager: DatabaseManager):
        self.config = config
        self.cache = deque(maxlen=config.cache_size)
        self.last_update = None
        self.last_cache_update = None
        self.error_count = 0
        self.success_count = 0

        # DB 매니저 인스턴스 (database.py에서 가져옴)
        self.db_manager = db_manager

    async def initialize(self):
        """수집기 초기화"""
        self.last_update = datetime.now()
        logging.info(f"{self.__class__.__name__} initialized")

    def _is_cache_valid(self) -> bool:
        """캐시 유효성 확인"""
        if not self.last_cache_update:
            return False

        elapsed_ms = (datetime.now() - self.last_cache_update).total_seconds() * 1000
        return elapsed_ms < self.config.cache_duration_ms

    async def _retry_operation(self, operation, *args, **kwargs):
        """
        작업 재시도 로직
        operation은 async 함수(또는 동기함수를 래핑한 코루틴)라고 가정.
        """
        for attempt in range(self.config.retry_attempts):
            try:
                result = await operation(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        return None

    @abstractmethod
    async def collect(self) -> Optional[MarketData]:
        """데이터 수집 추상 메서드"""
        pass

    async def _save_to_db(self, market_data: MarketData):
        """DB 저장 로직"""
        try:
            # MarketData 객체를 save_trade에 맞는 형식으로 변환
            trade_data = {
                "trade_type": "market_data",
                "symbol": "BTC-KRW",  # 기본값 또는 설정에서 가져옴
                "price": market_data.price,
                "volume": market_data.volume
            }
            # save_market_data 대신 save_trade 호출
            row_id = await self.db_manager.save_trade(trade_data)
            logging.info(f"[DB] market_data saved with row_id: {row_id}")
        except Exception as e:
            logging.error(f"Failed to save market_data to DB: {e}")


class PriceCollector(BaseCollector):
    """개선된 가격 데이터 수집기"""
    def __init__(self, config: CollectorConfig, db_manager: DatabaseManager):
        super().__init__(config, db_manager)
        self.recent_prices = deque(maxlen=100)
        # 가격은 REST API 위주로 처리 (실시간이 꼭 필요하면 별도 WebSocketManager 고려)

    async def collect(self) -> Optional[MarketData]:
        """가격 데이터 수집 메서드"""
        try:
            # 캐시 확인
            if self.cache and self._is_cache_valid():
                return self.cache[-1]

            # 현재가 조회
            current_price = await self._get_current_price()
            if not current_price:
                raise ValueError("Failed to fetch current price")

            # OHLCV 데이터 수집 (병렬 처리)
            # return_exceptions=True를 사용하여 부분 실패를 잡아냄
            results = await asyncio.gather(
                self._get_ohlcv("minute1", 10),
                self._get_ohlcv("minute60", 24),
                self._get_ohlcv("day", 30),
                return_exceptions=True
            )

            minute_data, hourly_data, daily_data = results

            # 각 결과에 대해 Exception 여부 확인
            if isinstance(minute_data, Exception):
                logging.error(f"Minute OHLCV fetch error: {minute_data}")
                minute_data = None
            if isinstance(hourly_data, Exception):
                logging.error(f"Hourly OHLCV fetch error: {hourly_data}")
                hourly_data = None
            if isinstance(daily_data, Exception):
                logging.error(f"Daily OHLCV fetch error: {daily_data}")
                daily_data = None

            # 가격 이력 업데이트
            self.recent_prices.append(current_price)

            # volume은 minute_data가 있다면 해당 시점의 거래량을 사용
            last_volume = 0
            if minute_data is not None and not minute_data.empty:
                last_volume = minute_data['volume'].iloc[-1]

            # 데이터 구조화
            market_data = MarketData(
                timestamp=datetime.now(),
                price=current_price,
                volume=last_volume
            )

            # 데이터 검증
            if not market_data.validate():
                raise ValueError("Invalid market data")

            # 캐시 업데이트
            self.cache.append(market_data)
            self.last_cache_update = datetime.now()
            self.success_count += 1

            # DB 저장
            await self._save_to_db(market_data)

            return market_data

        except Exception as e:
            self.error_count += 1
            logging.error(f"PriceCollector error: {e}")
            return None

    async def _get_current_price(self) -> Optional[float]:
        """현재가 조회 (동기함수를 비동기로 안전하게 호출)"""
        async def fetch_price() -> Optional[float]:
            # pyupbit.get_current_price는 동기 함수이므로 to_thread 사용
            return await asyncio.to_thread(pyupbit.get_current_price, self.config.ticker)

        for attempt in range(self.config.retry_attempts):
            try:
                price = await fetch_price()
                if price is not None:
                    return price
            except Exception as e:
                logging.warning(f"현재가 조회 시도 {attempt + 1} 실패: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        # 모든 시도 실패
        return None

    async def _get_ohlcv(self, interval: str, count: int):
        """OHLCV 데이터 조회 (동기함수를 비동기로 안전하게 호출)"""
        async def fetch_ohlcv():
            return await asyncio.to_thread(
                pyupbit.get_ohlcv,
                self.config.ticker,
                interval=interval,
                count=count
            )

        for attempt in range(self.config.retry_attempts):
            try:
                data = await fetch_ohlcv()
                if data is not None and not data.empty:
                    return data
            except Exception as e:
                logging.warning(f"OHLCV 조회 시도 {attempt + 1} 실패: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        # 모든 시도 실패
        return None

    def get_price_stats(self) -> Dict:
        """가격 통계 정보"""
        if not self.recent_prices:
            return {}

        prices = list(self.recent_prices)
        return {
            'current': prices[-1],
            'mean': sum(prices) / len(prices),
            'max': max(prices),
            'min': min(prices),
            'volatility': self._calculate_volatility(prices)
        }

    def _calculate_volatility(self, prices: List[float]) -> float:
        """변동성 계산"""
        if len(prices) < 2:
            return 0.0

        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] != 0:
                ret = (prices[i] - prices[i - 1]) / prices[i - 1]
                returns.append(ret)

        if not returns:
            return 0.0

        return (sum(r * r for r in returns) / len(returns)) ** 0.5


class OrderBookCollector(BaseCollector):
    """개선된 호가창 데이터 수집기 (웹소켓 기반 실시간 수집)"""
    def __init__(self, config: CollectorConfig, db_manager: DatabaseManager):
        super().__init__(config, db_manager)
        # WebSocketManager 초기화 (호가 수신 전용)
        self.websocket_manager = WebSocketManager("orderbook", [self.config.ticker])
        self.imbalance_history = deque(maxlen=100)

    async def collect(self) -> Optional[MarketData]:
        """호가창 데이터 실시간 수집 (웹소켓 사용)"""
        try:
            # 캐시 확인
            if self.cache and self._is_cache_valid():
                return self.cache[-1]

            # 웹소켓에서 최신 호가 데이터 수신
            orderbook = await self._get_orderbook()
            if not orderbook:
                raise ValueError("Failed to fetch orderbook")

            # 데이터 구조화
            market_data = MarketData(
                timestamp=datetime.now(),
                price=orderbook['orderbook_units'][0]['ask_price'],
                volume=sum(unit['ask_size'] for unit in orderbook['orderbook_units']),
                orderbook=orderbook
            )

            # 데이터 검증
            if not market_data.validate():
                raise ValueError("Invalid orderbook data")

            # 호가 불균형 분석
            imbalance = await self._analyze_imbalance(orderbook)
            self.imbalance_history.append(imbalance)

            # 캐시 업데이트
            self.cache.append(market_data)
            self.last_cache_update = datetime.now()
            self.success_count += 1

            # DB 저장
            await self._save_to_db(market_data)

            return market_data

        except Exception as e:
            self.error_count += 1
            logging.error(f"OrderBookCollector error: {e}")
            return None

    async def _get_orderbook(self) -> Optional[Dict]:
        """웹소켓에서 호가 데이터 1회 수신"""
        def _fetch_from_ws(manager: WebSocketManager):
            """
            manager.get()이 데이터를 받을 때까지 블로킹하므로,
            한 번만 읽고 반환. 예외 발생 시 None 또는 ValueError 처리.
            """
            data = manager.get()
            if not data:
                raise ValueError("No data received from WebSocket.")
            return data

        for attempt in range(self.config.retry_attempts):
            try:
                orderbook_data = await asyncio.to_thread(_fetch_from_ws, self.websocket_manager)
                if orderbook_data:
                    return orderbook_data
            except Exception as e:
                logging.warning(f"호가 조회 시도 {attempt + 1} 실패: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        # 모든 시도 실패
        return None

    async def _analyze_imbalance(self, orderbook: Dict) -> float:
        """호가 불균형 분석"""
        try:
            total_bid_size = sum(unit['bid_size'] for unit in orderbook['orderbook_units'])
            total_ask_size = sum(unit['ask_size'] for unit in orderbook['orderbook_units'])

            if (total_bid_size + total_ask_size) == 0:
                return 0.0

            # (매수총합 - 매도총합) / (매수총합 + 매도총합)
            return (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
        except Exception as e:
            logging.error(f"Imbalance analysis error: {e}")
            return 0.0

    def get_orderbook_stats(self) -> Dict:
        """호가창 통계 정보"""
        if not self.imbalance_history:
            return {}

        imbalances = list(self.imbalance_history)
        return {
            'current_imbalance': imbalances[-1],
            'mean_imbalance': sum(imbalances) / len(imbalances),
            'max_imbalance': max(imbalances),
            'min_imbalance': min(imbalances)
        }


class DataCollectionOrchestrator:
    """개선된 데이터 수집 오케스트레이터"""
    def __init__(self, config: CollectorConfig, db_path: str = "trading_history.db"):
        # DB 매니저를 생성 후, 각 Collector에게 주입
        from src.database import DatabaseConfig, ImprovedDatabaseManager
        db_config = DatabaseConfig(db_path=db_path)
        self.db_manager = ImprovedDatabaseManager(db_config)
        self.price_collector = PriceCollector(config, db_manager=self.db_manager)
        self.orderbook_collector = OrderBookCollector(config, db_manager=self.db_manager)
        self.collection_count = 0
        self.error_count = 0

    async def initialize_all(self):
        """모든 수집기 초기화"""
        await asyncio.gather(
            self.price_collector.initialize(),
            self.orderbook_collector.initialize()
        )
        logging.info("All collectors initialized")

    async def collect_all(self) -> Dict:
        """
        모든 데이터 수집.
        부분 실패가 있더라도 최대한 데이터를 모아서 반환하도록 개선.
        """
        try:
            # 병렬로 데이터 수집, return_exceptions=True로 부분 실패 허용
            results = await asyncio.gather(
                self.price_collector.collect(),
                self.orderbook_collector.collect(),
                return_exceptions=True
            )

            price_data, orderbook_data = results

            # 각 결과에 대해 예외 발생 여부 확인
            if isinstance(price_data, Exception):
                logging.error(f"Price collector failed: {price_data}")
                price_data = None
            if isinstance(orderbook_data, Exception):
                logging.error(f"Orderbook collector failed: {orderbook_data}")
                orderbook_data = None

            # 둘 다 실패했을 경우에만 전체 에러로 처리
            if not price_data and not orderbook_data:
                self.error_count += 1
                raise ValueError("Data collection failed for both price_data and orderbook_data")

            self.collection_count += 1

            return {
                "timestamp": datetime.now(),
                "price_data": price_data,
                "orderbook_data": orderbook_data,
                "collection_stats": self.get_collection_stats()
            }

        except Exception as e:
            self.error_count += 1
            logging.error(f"Data collection error: {e}")
            # 필요한 경우 여기서 예외 재발생
            raise

    def get_collection_stats(self) -> Dict:
        """수집 통계 정보"""
        return {
            "total_collections": self.collection_count,
            "error_count": self.error_count,
            "success_rate": (
                (self.collection_count - self.error_count) /
                max(1, self.collection_count)
            ) * 100,
            "price_collector_stats": {
                "success_count": self.price_collector.success_count,
                "error_count": self.price_collector.error_count
            },
            "orderbook_collector_stats": {
                "success_count": self.orderbook_collector.success_count,
                "error_count": self.orderbook_collector.error_count
            }
        }
class DataCollector(PriceCollector):
    """기본 데이터 수집기 (PriceCollector를 상속)"""
    def __init__(self, config: CollectorConfig, db_manager: DatabaseManager):
        super().__init__(config, db_manager)
        logging.info(f"DataCollector initialized for {config.ticker}")

# 모듈 내에서 임포트할 수 있도록 설정
__all__ = ["DataCollector", "PriceCollector", "OrderBookCollector", "CollectorConfig", "MarketData"]


# -------------------------------------------------------------------
# 멀티스레딩(또는 멀티프로세싱)을 활용한 예시:
# 여러 CollectorConfig에 대해 병렬적으로 collect_all()을 실행하고 싶다면,
# 아래처럼 concurrent.futures의 ThreadPoolExecutor 등을 사용할 수 있습니다.
# -------------------------------------------------------------------
def collect_data_in_thread(orchestrator: DataCollectionOrchestrator, loop: asyncio.AbstractEventLoop):
    """별도의 스레드에서 이벤트 루프를 돌려 수집 작업 실행"""
    asyncio.set_event_loop(loop)
    loop.run_until_complete(orchestrator.initialize_all())
    # 필요하다면 지속적으로 수집할 수도 있음 (여기서는 1회만)
    data = loop.run_until_complete(orchestrator.collect_all())
    return data

def run_multi_threaded_collectors(configs: List[CollectorConfig]):
    """예시: 여러 CollectorConfig(다른 거래소/마켓 등)에 대해 병렬 수집"""
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for cfg in configs:
            orchestrator = DataCollectionOrchestrator(cfg)
            loop = asyncio.new_event_loop()
            future = executor.submit(collect_data_in_thread, orchestrator, loop)
            futures.append(future)

        for f in futures:
            try:
                results.append(f.result())
            except Exception as e:
                logging.error(f"Threaded collection failed: {e}")
    return results


# 직접 실행 시 (테스트용) ---------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 단일 오케스트레이터 실행 예시
    single_config = CollectorConfig(ticker="KRW-BTC")
    orchestrator = DataCollectionOrchestrator(single_config, db_path="trading_history.db")

    async def main():
        await orchestrator.initialize_all()
        data = await orchestrator.collect_all()
        print("Collected Data:", data)

    asyncio.run(main())

    # 멀티스레딩 예시
    # multi_configs = [
    #     CollectorConfig(ticker="KRW-BTC"),
    #     CollectorConfig(ticker="KRW-ETH"),
    # ]
    # result_data = run_multi_threaded_collectors(multi_configs)
    # print("Multi-threaded collection results:", result_data)
