# src/real_time_analyzer.py

import asyncio
import logging
import time
import psutil
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional
from src.models.openai_client import OpenAIClient

# 만약 market_analyzer.py에서 특정 함수를 가져와야 한다면 아래처럼 import
# (실제 구현 내용에 맞춰 변경하세요)
try:
    from src.market_analyzer import get_current_liquidity_info
except ImportError:
    # 예: market_analyzer.py가 아직 없다면 예외처리
    def get_current_liquidity_info():
        """
        임시 함수: 실제 market_analyzer.py의 유동성 분석 함수를 대체
        프로젝트 상황에 맞춰 수정하세요.
        """
        # 예시로, 단순히 유동성 지표를 dict로 반환
        return {
            "liquidity_score": 0.8,
            "average_order_size": 1000.0  # 예시 (KRW 만원 단위 etc.)
        }


# 로깅 설정
logging.basicConfig(
    filename="real_time_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

@dataclass
class MarketData:
    """시장 데이터를 저장하는 데이터 클래스"""
    price: float
    volume: float
    trend: str

class RealTimeAnalyzer:
    def __init__(self):
        """ OpenAI O1 모델을 활용한 실시간 시장 분석 클래스 """
        self.openai_client = OpenAIClient()
        self.analysis_interval = 60  # 1분 단위 분석

    # --------------------------------------------------------
    # 1) 대량 주문이 시장에 미치는 영향 분석
    # --------------------------------------------------------
    def calculate_market_impact(
        self, 
        order_size: float, 
        average_order_size: float, 
        daily_volume: float
    ) -> float:
        """
        대량 주문의 시장 영향도를 계산합니다.
        
        Args:
            order_size (float): 이번에 발생한 특정 대량 주문의 크기
            average_order_size (float): 평균 주문 크기(시장 혹은 최근 기준)
            daily_volume (float): 하루 거래량(또는 분석기간 거래량)
        
        Returns:
            float: 0 ~ 1 범위의 시장 영향도 (예시)
        """
        # 예시 계산 로직 (단순화)
        # 1) 주문 크기가 평균 대비 몇 배인지
        size_ratio = order_size / max(1e-9, average_order_size)
        
        # 2) 해당 주문이 전체 거래량(daily_volume)에서 차지하는 비중
        volume_ratio = order_size / max(1e-9, daily_volume)
        
        # 3) 가중합 등을 통해 시장 영향도 도출 (단순 예시)
        impact_score = (size_ratio * 0.5) + (volume_ratio * 0.5)
        
        return impact_score

    # --------------------------------------------------------
    # 2) 매수·매도 압력 평가 (실시간 시장 움직임 예측)
    # --------------------------------------------------------
    def evaluate_market_pressure(self, orderbook_data: Dict) -> str:
        """
        호가창/주문장 정보를 기반으로 현재 시장의 매수·매도 압력을 평가합니다.
        
        Args:
            orderbook_data (Dict): 호가창 데이터(매도호가, 매수호가, 호가 잔량 등)
        
        Returns:
            str: "Bullish", "Bearish", "Neutral" 중 하나 예시
        """
        # 예시: 매수 잔량 vs 매도 잔량 비교
        total_bid_size = sum(unit["bid_size"] for unit in orderbook_data["orderbook_units"])
        total_ask_size = sum(unit["ask_size"] for unit in orderbook_data["orderbook_units"])
        
        if total_bid_size > total_ask_size * 1.2:
            return "Bullish"
        elif total_ask_size > total_bid_size * 1.2:
            return "Bearish"
        else:
            return "Neutral"

    # --------------------------------------------------------
    # 3) 주문장 분석(market_analyzer.py)과 연동하여 실시간 유동성 고려
    # --------------------------------------------------------
    def get_liquidity_info(self) -> Dict[str, float]:
        """
        market_analyzer.py 모듈과 연동하여, 현재 시장 유동성 정보를 가져옵니다.
        
        Returns:
            Dict[str, float]: 예) {"liquidity_score": 0.8, "average_order_size": 1000.0 ...}
        """
        return get_current_liquidity_info()

    # --------------------------------------------------------
    # 4) 실시간 시장 분석 (OpenAI + 다양한 분석 요소 통합)
    # --------------------------------------------------------
    def analyze_market(self, market_data: MarketData) -> str:
        """
        시장 데이터를 분석하여 트레이딩 전략을 생성하는 함수.
        
        Args:
            market_data (MarketData): 분석할 시장 데이터 객체
        
        Returns:
            str: OpenAI O1 모델을 통해 생성된 AI 기반 분석 결과
        """
        # (1) market_analyzer.py 연동: 실시간 유동성 등 가져오기
        liquidity_info = self.get_liquidity_info()  # {"liquidity_score": 0.8, "average_order_size": 1000.0, ...}
        liquidity_score = liquidity_info.get("liquidity_score", 0.5)
        average_order_size = liquidity_info.get("average_order_size", 1000.0)

        # (2) 대량 주문 발생 가정(예시)
        # 실제로는 외부에서 "가장 최근 거래" 등의 데이터를 수집해 order_size를 얻을 수 있음
        # 여기서는 임의 값으로 가정
        order_size = 3000.0  # 예: 3000만 원
        daily_volume = market_data.volume  # 일일 거래량 (단순히 MarketData.volume 이용 예시)

        impact_score = self.calculate_market_impact(order_size, average_order_size, daily_volume)
        
        # (3) 주문장 분석(매수·매도 압력)
        # 실제로는 OrderBookCollector에서 받은 orderbook_data를 파라미터로 넘겨야 함
        # 여기서는 샘플 dict
        sample_orderbook_data = {
            "orderbook_units": [
                {"bid_size": 10.0, "ask_size": 8.0},
                {"bid_size": 5.0,  "ask_size": 7.0},
            ]
        }
        market_pressure = self.evaluate_market_pressure(sample_orderbook_data)
        
        # (4) AI 프롬프트 구성
        prompt = (
            "Analyze the following market data with additional info:\n"
            f"- Price: {market_data.price}\n"
            f"- Volume (daily): {market_data.volume}\n"
            f"- Trend: {market_data.trend}\n"
            f"- Liquidity Score: {liquidity_score}\n"
            f"- Order Size: {order_size}\n"
            f"- Avg Order Size: {average_order_size}\n"
            f"- Impact Score: {impact_score:.4f}\n"
            f"- Market Pressure: {market_pressure}\n"
            "Discuss potential trading strategies, risk factors, and possible short-term market movement.\n"
        )
        # (5) OpenAI를 이용해 분석
        analysis_result = self.openai_client.generate_response(prompt)
        if not analysis_result:
            return "Analysis failed. Unable to retrieve AI response."

        return analysis_result

    async def run_real_time_analysis(self):
        """ 
        실시간 시장 분석을 수행하는 함수 (1분 간격으로 실행) 
        """
        logging.info("Starting real-time market analysis...")
        while True:
            market_data = self.get_latest_market_data()
            if market_data:
                analysis_result = self.analyze_market(market_data)
                self.store_analysis_result(analysis_result)
                logging.info(f"Market Analysis Result: {analysis_result}")
            else:
                logging.warning("No market data available. Skipping analysis.")
            
            await asyncio.sleep(min(60, max(1, self.analysis_interval * 1.0)))

    def get_latest_market_data(self) -> Optional[MarketData]:
        """
        최신 시장 데이터를 가져오는 함수.
        (실제 데이터 수집 시스템과 연동해야 함)
        
        Returns:
            Optional[MarketData]: 최신 시장 데이터 객체 (없을 시 None)
        """
        # TODO: 실제 시장 데이터(예: PriceCollector, DB, REST API 등)에서 가져오는 로직으로 교체
        # 여기서는 예시로 고정값을 반환
        return MarketData(
            price=42650.0,  # 현재가 (KRW 가정)
            volume=1.2e6,   # 일일 거래량
            trend="Upward"
        )

    def store_analysis_result(self, analysis_result: str):
        """
        분석 결과를 저장하는 함수.
        (실제 데이터베이스 저장 로직과 연동해야 함)
        
        Args:
            analysis_result (str): 분석된 시장 데이터
        """
        with open("market_analysis_log.txt", "a") as file:
            file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {analysis_result}\n")

    def monitor_system_resources(self) -> Dict[str, float]:
        """
        현재 CPU 및 메모리 사용량을 확인하는 함수.
        
        Returns:
            Dict[str, float]: CPU 및 메모리 사용률
        """
        return {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent
        }

# 실행 코드 (비동기 실행)
if __name__ == "__main__":
    analyzer = RealTimeAnalyzer()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(analyzer.run_real_time_analysis())
