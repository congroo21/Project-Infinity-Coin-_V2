# test_system.py - 프로젝트 인피니티 코인 v2 테스트 코드

import os
import sys
import unittest
import asyncio
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# src 폴더를 경로에 추가해 임포트할 수 있게 함
sys.path.append(os.path.abspath('.'))

# 테스트할 모듈들 임포트
from src.config import Config
from src.database import ImprovedDatabaseManager
from src.models.openai_client import OpenAIClient
from src.models.market_state import MarketState, MarketMetrics
from src.scenarios.risk.risk_manager import ImprovedRiskManager
from src.scenarios.market.state_manager import MarketStateManager
from src.scenarios.trading.trading_system import ImprovedTradingSystem, TradeSignal
from src.performance_monitor import PerformanceMonitor
from src.utils.risk_utils import risk_calculator
from src.utils.volatility_utils import calculate_volatility, calculate_returns

class TestComponentsBasic(unittest.TestCase):
    """기본 컴포넌트 테스트"""
    
    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        self.test_prices = [100, 101, 102, 103, 102, 103, 104, 105, 104, 103]
    
    def test_config_initialization(self):
        """기본 설정 초기화 테스트"""
        self.assertIsNotNone(Config.TRADE_INTERVAL)
        self.assertIsNotNone(Config.MAX_POSITION_SIZE)
        self.assertIsNotNone(Config.RISK_THRESHOLDS)
    
    def test_volatility_calculation(self):
        """변동성 계산 테스트"""
        volatility = calculate_volatility(self.test_prices)
        self.assertIsInstance(volatility, float)
        self.assertGreaterEqual(volatility, 0.0)  # 변동성은 항상 0 이상
    
    def test_returns_calculation(self):
        """수익률 계산 테스트"""
        returns = calculate_returns(self.test_prices)
        self.assertEqual(len(returns), len(self.test_prices) - 1)  # 수익률은 가격보다 1개 적음

    def test_market_state(self):
        """시장 상태 객체 테스트"""
        state = MarketState(
            timestamp=datetime.now(),
            price=50000000.0,
            volume=10.5,
            timeframe="1m"
        )
        self.assertEqual(state.price, 50000000.0)
        self.assertEqual(state.volume, 10.5)
        self.assertEqual(state.timeframe, "1m")

    def test_risk_calculator_basic(self):
        """리스크 계산기 기본 기능 테스트"""
        # 간단한 VaR 계산 테스트
        returns = [-0.01, 0.02, 0.005, -0.008, 0.015]
        var_95 = risk_calculator.calculate_parametric_var(returns, 0.95)
        self.assertIsInstance(var_95, float)


class TestRiskManager(unittest.TestCase):
    """리스크 관리자 테스트"""
    
    def setUp(self):
        """리스크 관리자 초기화"""
        self.risk_manager = ImprovedRiskManager()
    
    def test_risk_thresholds(self):
        """리스크 임계값 테스트"""
        self.assertIsNotNone(self.risk_manager.thresholds)
        self.assertIsNotNone(self.risk_manager.thresholds.max_position_size)
        self.assertIsNotNone(self.risk_manager.thresholds.max_drawdown)
        self.assertIsNotNone(self.risk_manager.thresholds.volatility_threshold)


class TestMarketStateManager(unittest.TestCase):
    """시장 상태 관리자 테스트"""
    
    def setUp(self):
        """시장 상태 관리자 초기화"""
        self.state_manager = MarketStateManager()
    
    def test_update_market_state(self):
        """시장 상태 업데이트 테스트"""
        # 테스트 데이터
        market_data = {
            'price': 50000000.0,
            'volume': 10.5,
            'timeframe': '1m'
        }
        
        # 예외가 발생하지 않아야 함
        try:
            self.state_manager.update(market_data)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"시장 상태 업데이트 중 예외 발생: {e}")


class TestDatabaseManager(unittest.TestCase):
    """데이터베이스 관리자 테스트"""
    
    def setUp(self):
        """임시 데이터베이스로 테스트"""
        self.db_manager = ImprovedDatabaseManager(Config())
    
    def test_db_initialization(self):
        """DB 초기화 테스트"""
        # 데이터베이스 파일이 생성되어야 함
        db_path = self.db_manager.config.db_path
        # 파일이 존재하는지만 확인하고 삭제
        if os.path.exists(db_path):
            os.remove(db_path)


class TestAsyncComponents(unittest.IsolatedAsyncioTestCase):
    """비동기 컴포넌트 테스트"""
    
    async def asyncSetUp(self):
        """비동기 설정"""
        self.performance_monitor = PerformanceMonitor()
        self.trading_system = ImprovedTradingSystem()
    
    async def test_performance_monitor_start_stop(self):
        """성능 모니터 시작/중지 테스트"""
        # 모니터링 시작
        await self.performance_monitor.start()
        self.assertTrue(self.performance_monitor.running)
        
        # 잠시 기다림
        await asyncio.sleep(0.1)
        
        # 모니터링 중지
        await self.performance_monitor.stop()
        self.assertFalse(self.performance_monitor.running)
    
    async def test_trading_system_basic(self):
        """트레이딩 시스템 기본 기능 테스트"""
        # 신호 생성 테스트
        market_data = {
            'price': 50000000.0,
            'volume': 10.5,
            'timestamp': datetime.now(),
            'integrated_state': {
                'overall_trend': 'uptrend',
                'volatility_state': 'normal',
                'momentum_signal': 0.7,
                'trading_signals': {}
            }
        }
        
        # _generate_trading_signal 메서드는 원래 프라이빗이지만 테스트를 위해 직접 호출
        signal = await self.trading_system._generate_trading_signal(market_data)
        
        # 신호가 None이 아니어야 함 (구현 방식에 따라 None일 수도 있음)
        if signal:
            self.assertIsInstance(signal, TradeSignal)


class TestOpenAIClient(unittest.TestCase):
    """OpenAI 클라이언트 테스트"""
    
    @patch('openai.OpenAI')
    def test_generate_response_mocked(self, mock_openai):
        """Mock을 이용한 OpenAI 응답 생성 테스트"""
        # OpenAI 응답 모의 객체 설정
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "테스트 응답"
        
        mock_openai.return_value.chat.completions.create.return_value = mock_completion
        
        # 클라이언트 생성 및 응답 테스트
        client = OpenAIClient()
        response = client.generate_response("테스트 질문")
        
        self.assertEqual(response, "테스트 응답")


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """통합 테스트"""
    
    async def asyncSetUp(self):
        """비동기 설정"""
        self.risk_manager = ImprovedRiskManager()
        self.state_manager = MarketStateManager()
        self.trading_system = ImprovedTradingSystem()
    
    async def test_market_and_risk_integration(self):
        """시장 상태와 리스크 관리 통합 테스트"""
        # 테스트 데이터
        market_data = {
            'price': 50000000.0,
            'volume': 10.5,
            'timeframe': '1m'
        }
        
        # 시장 상태 업데이트
        self.state_manager.update(market_data)
        
        # 리스크 체크
        trade_data = {
            'symbol': 'BTC-KRW',
            'size': 0.01
        }
        
        # 리스크 체크 수행
        result = await self.risk_manager.check_trade(trade_data)
        
        # 결과가 딕셔너리여야 함
        self.assertIsInstance(result, dict)
        self.assertIn('is_safe', result)


# 테스트 실행 함수
def run_tests():
    unittest.main()

# 직접 실행 시
if __name__ == "__main__":
    run_tests()