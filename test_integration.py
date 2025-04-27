# test_integration.py - 프로젝트 인피니티 코인 v2 통합 테스트

import os
import sys
import unittest
import asyncio
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# src 폴더를 경로에 추가
sys.path.append(os.path.abspath('.'))

# 테스트 데이터 생성 함수
def create_market_data():
    """테스트용 시장 데이터 생성"""
    return {
        'price': 50000000.0,  # 5천만원
        'volume': 15.5,
        'timeframe': '1m',
        'timestamp': datetime.now(),
        'integrated_state': {
            'overall_trend': 'uptrend',
            'volatility_state': 'normal',
            'momentum_signal': 0.65,
            'market_condition': 'normal'
        }
    }

def create_orderbook_data():
    """테스트용 호가창 데이터 생성"""
    return {
        'orderbook_units': [
            {'ask_price': 50100000.0, 'ask_size': 0.5, 'bid_price': 49900000.0, 'bid_size': 0.8},
            {'ask_price': 50200000.0, 'ask_size': 1.2, 'bid_price': 49800000.0, 'bid_size': 1.5},
            {'ask_price': 50300000.0, 'ask_size': 2.0, 'bid_price': 49700000.0, 'bid_size': 1.0}
        ]
    }


class TestMarketAnalyzerIntegration(unittest.IsolatedAsyncioTestCase):
    """시장 분석기 통합 테스트"""
    
    async def asyncSetUp(self):
        """비동기 설정"""
        # 필요한 모듈 임포트 (클래스 내에서 필요할 때 임포트)
        from src.market_analyzer import MultiTimeframeMarketAnalyzer
        
        self.market_analyzer = MultiTimeframeMarketAnalyzer()
        
        # 테스트 데이터 설정
        self.market_data = create_market_data()
        
        # 초기 데이터 설정
        for tf in self.market_analyzer.timeframes:
            self.market_analyzer.data_stores[tf]['prices'] = [
                self.market_data['price'] * (1 + i * 0.001) for i in range(-50, 50)
            ]
            self.market_analyzer.data_stores[tf]['volumes'] = [
                self.market_data['volume'] * (1 + i * 0.01) for i in range(-50, 50)
            ]
    
    async def test_integrated_analysis(self):
        """통합 분석 테스트"""
        # 캐시 무효화
        self.market_analyzer.cached_analysis = None
        self.market_analyzer.last_cache_update = None
        
        # 통합 분석 실행
        analysis = await self.market_analyzer.get_integrated_analysis()
        
        # 결과 검증
        self.assertIsNotNone(analysis)
        self.assertIn('overall_trend', analysis)
        self.assertIn('volatility_state', analysis)
        self.assertIn('momentum_signal', analysis)
        self.assertIn('market_condition', analysis)


class TestRiskManagerIntegration(unittest.IsolatedAsyncioTestCase):
    """리스크 관리자 통합 테스트"""
    
    async def asyncSetUp(self):
        """비동기 설정"""
        from src.scenarios.risk.risk_manager import ImprovedRiskManager
        from src.database import ImprovedDatabaseManager
        
        # 데이터베이스 관리자 mock 생성
        self.db_manager = MagicMock(spec=ImprovedDatabaseManager)
        self.db_manager.save_risk_event = AsyncMock()
        self.db_manager.save_risk_state = AsyncMock()
        
        # 리스크 관리자 생성 (모의 DB 사용)
        self.risk_manager = ImprovedRiskManager()
        self.risk_manager.db = self.db_manager
        
        # 모니터링 작업을 위한 실행 플래그 설정
        self.risk_manager.running = True
    
    async def test_risk_check_workflow(self):
        """리스크 체크 워크플로우 테스트"""
        # 테스트 거래 데이터
        trade_data = {
            'symbol': 'BTC-KRW',
            'size': 0.05,  # MAX_POSITION_SIZE보다 작아야 함
            'price': 50000000
        }
        
        # 리스크 체크 실행
        result = await self.risk_manager.check_trade(trade_data)
        
        # 결과 검증
        self.assertIsInstance(result, dict)
        self.assertIn('is_safe', result)
        
        # 안전한 거래여야 함
        if not result.get('is_safe', False):
            self.fail(f"거래가 안전하지 않음: {result.get('reason', '이유 없음')}")
    
    async def test_position_update(self):
        """포지션 업데이트 테스트"""
        # 테스트 포지션 데이터
        position_data = {
            'symbol': 'BTC-KRW',
            'size': 0.03,
            'price': 50000000
        }
        
        # 포지션 업데이트 실행
        await self.risk_manager.update_position(position_data)
        
        # 상태 검증
        self.assertIn('BTC-KRW', self.risk_manager.state.current_positions)
        self.assertEqual(
            self.risk_manager.state.current_positions['BTC-KRW']['size'],
            position_data['size']
        )
    
    async def asyncTearDown(self):
        """테스트 종료 작업"""
        # 실행 중이었던 작업 중지
        self.risk_manager.running = False


class TestTradingSystemIntegration(unittest.IsolatedAsyncioTestCase):
    """트레이딩 시스템 통합 테스트"""
    
    async def asyncSetUp(self):
        """비동기 설정"""
        from src.scenarios.trading.trading_system import ImprovedTradingSystem
        
        self.trading_system = ImprovedTradingSystem()
        
        # 테스트 데이터
        self.market_data = {
            'price': 50000000.0,
            'volume': 10.5,
            'timestamp': datetime.now(),
            'integrated_state': {
                'overall_trend': 'uptrend',
                'volatility_state': 'normal',
                'momentum_signal': 0.7,
                'market_condition': 'stable_trending'
            }
        }
    
    async def test_trade_signal_generation(self):
        """거래 신호 생성 테스트"""
        # 거래 신호 생성
        signal = await self.trading_system._generate_trading_signal(self.market_data)
        
        # 결과 검증
        if signal:
            self.assertIsNotNone(signal.symbol)
            self.assertIsNotNone(signal.action)
            self.assertIsNotNone(signal.price)
            self.assertIsNotNone(signal.size)
        else:
            # 신호가 없는 경우는 정상이지만 로그 기록
            logging.info("거래 신호가 생성되지 않았습니다(정상)")
    
    async def test_trade_execution_mock(self):
        """모의 거래 실행 테스트"""
        from src.scenarios.trading.trading_system import TradeSignal
        
        # 테스트용 신호 생성
        test_signal = TradeSignal(
            timestamp=datetime.now(),
            symbol="BTC-KRW",
            action="buy",
            price=50000000.0,
            size=0.01,
            timeframe="1m",
            confidence=0.8,
            signal_source="test",
            risk_score=0.3
        )
        
        # 거래 실행
        result = await self.trading_system._execute_trade(test_signal)
        
        # 결과 검증
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'success'))
        if not result.success:
            logging.warning(f"거래 실행 실패: {result.error_message}")


# Mock 클래스 (비동기 mock을 위한 헬퍼)
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


# 직접 실행 시
if __name__ == "__main__":
    unittest.main()