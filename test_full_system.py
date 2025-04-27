# test_full_system.py - 프로젝트 인피니티 코인 v2 전체 시스템 테스트

import os
import sys
import asyncio
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_system.log"),
        logging.StreamHandler()
    ]
)

# src 폴더를 경로에 추가
sys.path.append(os.path.abspath('.'))

# 테스트 환경 설정 (실제 시스템 모듈)
from src.config import Config
from src.data_collector import CollectorConfig, DataCollectionOrchestrator
from src.market_analyzer import MultiTimeframeMarketAnalyzer
from src.scenarios.risk.risk_manager import ImprovedRiskManager
from src.scenarios.trading.trading_system import ImprovedTradingSystem
from src.performance_monitor import PerformanceMonitor
from src.utils.risk_dashboard import risk_dashboard
from src.database import ImprovedDatabaseManager

# 글로벌 변수
shutdown_requested = False

# 테스트 데이터 생성 함수들
def create_sample_market_data():
    """샘플 시장 데이터 생성"""
    prices = [50000000.0 * (1 + i * 0.001) for i in range(-20, 20)]
    volumes = [10.5 * (1 + i * 0.01) for i in range(-20, 20)]
    
    return {
        'prices': prices,
        'volumes': volumes,
        'timestamp': datetime.now(),
        'price': prices[-1],  # 현재가
        'volume': volumes[-1],  # 현재 거래량
        'timeframe': '1m',
    }

def mock_data_collection():
    """데이터 수집을 모의하는 함수"""
    return {
        'price_data': {
            'price': 50000000.0,
            'volume': 10.5,
            'timestamp': datetime.now()
        },
        'orderbook_data': {
            'orderbook_units': [
                {'ask_price': 50100000.0, 'ask_size': 0.5, 'bid_price': 49900000.0, 'bid_size': 0.8},
                {'ask_price': 50200000.0, 'ask_size': 1.2, 'bid_price': 49800000.0, 'bid_size': 1.5}
            ]
        }
    }

# 단일 컴포넌트 mock 생성 함수
def create_mock_component(component_type):
    """단일 컴포넌트 mock 생성"""
    mock = MagicMock(spec=component_type)
    
    # 비동기 함수에 대한 특별 처리
    if hasattr(component_type, 'start'):
        mock.start = AsyncMock()
    if hasattr(component_type, 'stop'):
        mock.stop = AsyncMock()
    if hasattr(component_type, 'collect_all'):
        mock.collect_all = AsyncMock(return_value=mock_data_collection())
    if hasattr(component_type, 'get_integrated_analysis'):
        mock.get_integrated_analysis = AsyncMock(return_value={
            'overall_trend': 'uptrend',
            'volatility_state': 'normal',
            'momentum_signal': 0.7,
            'market_condition': 'stable_trending',
            'risk_metrics': {
                'volatility': 0.02,
                'var_95': 0.01,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.03
            }
        })
    if hasattr(component_type, 'get_risk_status'):
        mock.get_risk_status = MagicMock(return_value={
            'warning_level': 'normal',
            'risk_level': 'normal',
            'active_alerts': [],
            'risk_metrics': {
                'volatility': 0.02,
                'var_95': 0.01
            }
        })
    
    return mock

# AsyncMock 클래스 (비동기 mock을 위한 헬퍼)
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

# 시스템 테스트 함수
async def test_full_system_mock():
    """Mock을 사용한 전체 시스템 테스트"""
    logging.info("=== 전체 시스템 테스트 시작 (Mock 사용) ===")
    
    # 필요한 컴포넌트 생성 (실제 객체 또는 Mock)
    components = {}
    
    # DB 관리자 (실제 객체 사용 - 임시 데이터베이스)
    db_config = Config()
    db_config.DB_PATH = "test_trading.db"  # 테스트용 DB 경로
    components['db_manager'] = ImprovedDatabaseManager(db_config)
    
    # 다른 컴포넌트들 (Mock 사용)
    components['data_orchestrator'] = create_mock_component(DataCollectionOrchestrator)
    components['market_analyzer'] = create_mock_component(MultiTimeframeMarketAnalyzer)
    components['risk_manager'] = create_mock_component(ImprovedRiskManager)
    components['trading_system'] = create_mock_component(ImprovedTradingSystem)
    components['performance_monitor'] = create_mock_component(PerformanceMonitor)
    
    # 간략화된 트레이딩 루프
    try:
        logging.info("트레이딩 시뮬레이션 시작")
        for _ in range(5):  # 5회만 반복
            # 1. 데이터 수집
            collected_data = await components['data_orchestrator'].collect_all()
            logging.info(f"1. 데이터 수집 완료: {collected_data['price_data']['price']}")
            
            # 2. 시장 분석
            analysis_result = await components['market_analyzer'].get_integrated_analysis()
            logging.info(f"2. 시장 분석 완료: 추세={analysis_result['overall_trend']}")
            
            # 3. 리스크 체크
            risk_status = components['risk_manager'].get_risk_status()
            logging.info(f"3. 리스크 체크 완료: 상태={risk_status['risk_level']}")
            
            # 4. 리스크 대시보드 업데이트
            risk_dashboard.update_metrics({
                'market_analyzer': analysis_result.get('risk_metrics', {}),
                'risk_manager': risk_status.get('risk_metrics', {})
            })
            dashboard_metrics = risk_dashboard.get_consolidated_metrics()
            logging.info(f"4. 리스크 대시보드 업데이트: 점수={dashboard_metrics.get('risk_score', 0)}")
            
            # 5. 트레이딩 로직 (간단한 예시)
            market_data = {
                'price': collected_data['price_data']['price'],
                'timestamp': collected_data['price_data']['timestamp'],
                'integrated_state': analysis_result,
                'risk_status': risk_status
            }
            
            # 트레이딩 신호 생성 (Mock이므로 함수 호출만 수행)
            components['trading_system']._generate_trading_signal(market_data)
            logging.info("5. 트레이딩 신호 생성 완료")
            
            # 짧은 대기 (실제로는 Config.TRADE_INTERVAL에 따라 조정)
            await asyncio.sleep(0.01)
        
        logging.info("트레이딩 시뮬레이션 완료")
        
    except Exception as e:
        logging.error(f"테스트 실행 중 오류 발생: {e}")
    finally:
        # 정리 작업
        logging.info("테스트 정리 작업 수행 중...")
        
        # 컴포넌트 종료
        for name, component in components.items():
            if hasattr(component, 'stop') and callable(component.stop):
                try:
                    await component.stop()
                    logging.info(f"{name} 컴포넌트 정상 종료")
                except Exception as e:
                    logging.error(f"{name} 컴포넌트 종료 중 오류: {e}")
    
    logging.info("=== 전체 시스템 테스트 완료 ===")
    
    # 테스트 DB 파일 삭제
    if os.path.exists(db_config.DB_PATH):
        os.remove(db_config.DB_PATH)
        logging.info(f"테스트 데이터베이스 파일 삭제: {db_config.DB_PATH}")


# 실제 시스템 테스트 (최소한의 실제 구성 요소 사용)
async def test_minimal_real_system():
    """최소한의 실제 구성 요소를 사용한 테스트"""
    logging.info("=== 최소한의 실제 시스템 테스트 시작 ===")
    
    components = {}
    
    try:
        # 데이터베이스 매니저 초기화 (실제 객체)
        db_config = Config()
        db_config.DB_PATH = "test_real_system.db"
        components['db_manager'] = ImprovedDatabaseManager(db_config)
        
        # 리스크 관리자 초기화 (실제 객체)
        components['risk_manager'] = ImprovedRiskManager()
        await components['risk_manager'].start_monitoring()
        
        # 성능 모니터 초기화 (실제 객체)
        components['performance_monitor'] = PerformanceMonitor()
        await components['performance_monitor'].start()
        
        # 다른 컴포넌트들은 Mock 사용
        components['data_orchestrator'] = create_mock_component(DataCollectionOrchestrator)
        components['market_analyzer'] = create_mock_component(MultiTimeframeMarketAnalyzer)
        components['trading_system'] = create_mock_component(ImprovedTradingSystem)
        
        # 간략화된 테스트 실행
        logging.info("실제 구성 요소를 포함한 시뮬레이션 시작")
        
        # 거래 시뮬레이션
        for i in range(3):  # 3회만 반복
            logging.info(f"반복 {i+1}/3 시작")
            
            # 데이터 수집 (Mock)
            collected_data = await components['data_orchestrator'].collect_all()
            
            # 포지션 업데이트
            position_data = {
                'symbol': 'BTC-KRW',
                'size': 0.01 * (i + 1),  # 점점 크기 증가
                'price': collected_data['price_data']['price']
            }
            await components['risk_manager'].update_position(position_data)
            logging.info(f"포지션 업데이트: {position_data['symbol']} - 크기: {position_data['size']}")
            
            # 거래 체크
            trade_data = {
                'symbol': 'BTC-KRW',
                'size': 0.01,
            }
            result = await components['risk_manager'].check_trade(trade_data)
            logging.info(f"거래 체크 결과: {result}")
            
            # 성능 기록
            components['performance_monitor'].record_message(latency=0.05)
            
            # 성능 통계 확인
            stats = components['performance_monitor'].get_stats()
            logging.info(f"성능 통계: {stats}")
            
            # 짧은 대기
            await asyncio.sleep(0.1)
        
        # 리스크 상태 최종 확인
        risk_status = components['risk_manager'].get_risk_status()
        logging.info(f"최종 리스크 상태: {risk_status}")
        
        logging.info("실제 구성 요소를 포함한 시뮬레이션 완료")
        
    except Exception as e:
        logging.error(f"실제 시스템 테스트 중 오류 발생: {e}")
    finally:
        # 컴포넌트 종료
        for name, component in components.items():
            if hasattr(component, 'stop') and callable(component.stop):
                try:
                    await component.stop()
                    logging.info(f"{name} 컴포넌트 정상 종료")
                except Exception as e:
                    logging.error(f"{name} 컴포넌트 종료 중 오류: {e}")
    
    logging.info("=== 최소한의 실제 시스템 테스트 완료 ===")
    
    # 테스트 DB 파일 삭제
    if os.path.exists(db_config.DB_PATH):
        os.remove(db_config.DB_PATH)
        logging.info(f"테스트 데이터베이스 파일 삭제: {db_config.DB_PATH}")


# 테스트 실행 함수
async def run_all_tests():
    """모든 테스트 실행"""
    logging.info("테스트 시작")
    
    # 1. Mock 기반 전체 시스템 테스트
    await test_full_system_mock()
    
    # 2. 최소한의 실제 구성 요소를 사용한 테스트
    await test_minimal_real_system()
    
    logging.info("모든 테스트 완료")


# 직접 실행 시
if __name__ == "__main__":
    # 비동기 이벤트 루프 실행
    asyncio.run(run_all_tests())