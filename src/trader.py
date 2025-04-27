import sys
import os
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any
import psutil

# openai_client.py가 있는 models 폴더를 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

from openai_client import OpenAIClient  # ✅ 경로 문제 수정

# 로깅 설정
logging.basicConfig(
    filename="trader.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

@dataclass
class TradeDecision:
    """ 트레이딩 결정을 저장하는 데이터 클래스 """
    symbol: str
    price: float
    volume: float
    trend: str
    timestamp: str
    decision: str

class Trader:
    def __init__(self):
        """ OpenAI O1 모델을 활용한 트레이딩 시스템 """
        self.openai_client = OpenAIClient()
        self.trade_interval = 60  # 1분 단위 트레이딩 실행
        self.previous_trade_decisions = {}

    def execute_trade(self, market_data: TradeDecision) -> str:
        """
        시장 데이터를 기반으로 트레이딩 결정을 수행하는 함수.
        
        Args:
            market_data (TradeDecision): 분석할 시장 데이터 객체
        
        Returns:
            str: OpenAI O1 모델을 통해 생성된 매매 결정 결과
        """
        prompt = (
            "Based on the following market data, decide whether to Buy, Sell, or Hold:\n"
            f"Symbol: {market_data.symbol}\n"
            f"Price: {market_data.price}\n"
            f"Volume: {market_data.volume}\n"
            f"Trend: {market_data.trend}\n"
            f"Timestamp: {market_data.timestamp}\n"
        )
        decision = self.openai_client.generate_response(prompt)
        
        if not decision:
            return "Trade decision failed. Unable to retrieve AI response."
        
        return decision

    async def run_trading_system(self):
        """ 
        실시간 트레이딩 시스템을 수행하는 함수 (1분 간격으로 실행) 
        """
        logging.info("Starting real-time trading system...")
        while True:
            market_data = self.get_latest_market_data()
            if market_data:
                trade_decision = self.execute_trade(market_data)
                if self.should_execute_trade(market_data.symbol, trade_decision):
                    self.store_trade_decision(market_data, trade_decision)
                    self.perform_trade_action(market_data, trade_decision)
                    logging.info(f"Trade Decision: {trade_decision}")
                else:
                    logging.info("No significant market change detected. Skipping trade execution.")
            else:
                logging.warning("No market data available. Skipping trade execution.")
            
            await asyncio.sleep(self.trade_interval)  # 1분 대기 후 다시 실행

    def get_latest_market_data(self) -> TradeDecision:
        """
        최신 시장 데이터를 가져오는 함수.
        (실제 데이터 수집 시스템과 연동해야 함)
        
        Returns:
            TradeDecision: 최신 시장 데이터 객체
        """
        # TODO: 실제 시장 데이터 수집 로직으로 교체 필요
        return TradeDecision(
            symbol="BTC/USD",
            price=45000,
            volume=1.5e6,
            trend="Bullish",
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            decision="Hold"
        )

    def should_execute_trade(self, symbol: str, new_decision: str) -> bool:
        """
        이전 트레이드 결정과 비교하여 변화가 있을 경우에만 실행
        
        Args:
            symbol (str): 트레이딩 대상 자산의 심볼
            new_decision (str): 새로운 트레이딩 결정
        
        Returns:
            bool: 변화가 있으면 True, 없으면 False
        """
        if symbol not in self.previous_trade_decisions or self.previous_trade_decisions[symbol] != new_decision:
            self.previous_trade_decisions[symbol] = new_decision
            return True
        return False

    def store_trade_decision(self, market_data: TradeDecision, trade_decision: str):
        """
        트레이드 결정 결과를 저장하는 함수.
        (실제 데이터베이스 저장 로직과 연동해야 함)
        
        Args:
            market_data (TradeDecision): 트레이드 대상 시장 데이터
            trade_decision (str): 분석된 트레이드 결정 결과
        """
        log_entry = (
            f"{market_data.timestamp} - {market_data.symbol} - "
            f"Price: {market_data.price}, Volume: {market_data.volume}, Trend: {market_data.trend} - "
            f"Trade Decision: {trade_decision}\n"
        )
        with open("trade_log.txt", "a") as file:
            file.write(log_entry)

    def perform_trade_action(self, market_data: TradeDecision, trade_decision: str):
        """
        실제 트레이드 실행을 담당하는 함수.
        
        Args:
            market_data (TradeDecision): 트레이드 대상 시장 데이터
            trade_decision (str): AI가 결정한 트레이딩 액션 (Buy, Sell, Hold)
        """
        if trade_decision.lower() == "buy":
            logging.info(f"Executing BUY order for {market_data.symbol} at {market_data.price}")
        elif trade_decision.lower() == "sell":
            logging.info(f"Executing SELL order for {market_data.symbol} at {market_data.price}")
        else:
            logging.info(f"Holding position for {market_data.symbol}")

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
    trader = Trader()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(trader.run_trading_system())
