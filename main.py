# main.py

import os
import asyncio
import logging
import numpy as np
import signal
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# 환경 변수 및 설정 로드
from dotenv import load_dotenv
from src.config import Config

# 데이터 수집 및 분석
from src.data_collector import DataCollectionOrchestrator, CollectorConfig
from src.market_analyzer import MultiTimeframeMarketAnalyzer
from src.database import ImprovedDatabaseManager

# 시나리오 관리
from src.scenarios.market.state_manager import MarketStateManager
from src.scenarios.trading.scenario_generator import ScenarioGenerator, ScenarioConfig
from src.scenarios.trading.trading_system import ImprovedTradingSystem
from src.scenarios.risk.risk_manager import ImprovedRiskManager
from src.scenarios.monitoring.performance import ScenarioPerformanceMonitor

# 실시간 시스템
from src.real_time_analyzer import RealTimeManager
from src.memory_manager import MemoryManager, MemoryConfig
from src.websocket_manager import WebSocketManager, WebSocketConfig
from src.performance_monitor import PerformanceMonitor

# 뉴스 및 소셜 미디어 분석
from src.analyzers.news_analyzer import EnhancedNewsAnalyzer
from src.analyzers.reddit_analyzer import RedditAnalyzer
from src.collectors.news_collector import NewsCollector

# 온체인 데이터 분석 (새로 추가된 모듈)
from src.collectors.onchain_collector import OnchainCollector, BlockchainConfig
from src.analyzers.onchain_analyzer import OnchainAnalyzer, OnchainAnalysisConfig
from src.analyzers.transaction_analyzer import TransactionAnalyzer
from src.scenarios.models.blockchain_integration import BlockchainIntegration, initialize_blockchain_integration

# 예외 처리
from src.exceptions import (
    ValidationError, InsufficientDataError, MarketDataError,
    SystemResourceError, TradingSystemError, ConfigurationError,
    EmergencyShutdownError, SystemEmergencyError
)

# 리스크 유틸리티 및 대시보드 가져오기 (새로 추가)
from src.utils.risk_utils import risk_calculator
from src.utils.risk_dashboard import risk_dashboard

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로그 파일 경로
    log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("로깅 설정 완료")

# 정상 종료 처리
def handle_shutdown(signum, frame):
    """시그널 핸들러 (정상 종료 처리)"""
    logging.info("종료 신호 수신. 정상 종료 시작...")
    global shutdown_requested
    shutdown_requested = True

# 온체인 데이터 분석 기능 초기화 및 실행
async def initialize_onchain_analysis():
    """온체인 데이터 분석 기능 초기화"""
    try:
        # 데이터베이스 매니저 초기화
        db_manager = ImprovedDatabaseManager()
        
        # 블록체인 통합 모듈 초기화
        # 이더리움 메인넷 RPC URL (인프라나 알케미 등의 API 키 필요)
        rpc_url = os.getenv("ETHEREUM_RPC_URL", "https://mainnet.infura.io/v3/YOUR_API_KEY")
        blockchain_integration = await initialize_blockchain_integration(
            db_manager=db_manager,
            rpc_url=rpc_url,
            chain_id=1  # 이더리움 메인넷
        )
        
        # 통합 프로세스 시작
        asyncio.create_task(blockchain_integration.start_integration())
        
        logging.info("온체인 분석 시스템 초기화 완료")
        return blockchain_integration
        
    except Exception as e:
        logging.error(f"온체인 분석 초기화 오류: {e}")
        return None

# 구성 요소 초기화
async def initialize_components():
    """모든 구성 요소 초기화"""
    try:
        components = {}
        
        # 데이터베이스 매니저 초기화
        db_manager = ImprovedDatabaseManager()
        components['db_manager'] = db_manager
        
        # 데이터 수집기 초기화
        collector_config = CollectorConfig(
            ticker="KRW-BTC",
            update_interval=Config.ANALYSIS_INTERVAL
        )
        data_orchestrator = DataCollectionOrchestrator(collector_config, db_path=Config.DB_PATH)
        await data_orchestrator.initialize_all()
        components['data_orchestrator'] = data_orchestrator
        
        # 시장 분석기 초기화
        market_analyzer = MultiTimeframeMarketAnalyzer()
        asyncio.create_task(market_analyzer.start())
        components['market_analyzer'] = market_analyzer
        
        # 메모리 관리자 초기화
        memory_config = MemoryConfig(
            warning_threshold_mb=1000,
            critical_threshold_mb=1500
        )
        memory_manager = MemoryManager(memory_config)
        asyncio.create_task(memory_manager.start_monitoring())
        components['memory_manager'] = memory_manager
        
        # 성능 모니터링 초기화
        performance_monitor = PerformanceMonitor()
        await performance_monitor.start()
        components['performance_monitor'] = performance_monitor
        
        # 시나리오 모듈 초기화
        market_state_manager = MarketStateManager()
        scenario_config = ScenarioConfig()
        scenario_generator = ScenarioGenerator(scenario_config)
        components['market_state_manager'] = market_state_manager
        components['scenario_generator'] = scenario_generator
        
        # 리스크 관리자 초기화
        risk_manager = ImprovedRiskManager()
        asyncio.create_task(risk_manager.start_monitoring())
        components['risk_manager'] = risk_manager
        
        # 트레이딩 시스템 초기화
        trading_system = ImprovedTradingSystem()
        components['trading_system'] = trading_system
        
        # 뉴스 및 소셜 미디어 분석 모듈 초기화
        news_collector = NewsCollector({'update_interval': 300})
        await news_collector.start_collection()
        components['news_collector'] = news_collector
        
        news_analyzer = EnhancedNewsAnalyzer()
        components['news_analyzer'] = news_analyzer
        
        # 웹소켓 매니저 초기화
        ws_config = WebSocketConfig(
            uri="wss://api.upbit.com/websocket/v1",
            ping_interval=0.02
        )
        ws_manager = WebSocketManager(ws_config)
        await ws_manager.start()
        components['ws_manager'] = ws_manager
        
        # 온체인 데이터 분석 시스템 초기화 (새로 추가)
        blockchain_integration = await initialize_onchain_analysis()
        components['blockchain_integration'] = blockchain_integration
        
        logging.info("모든 구성 요소 초기화 완료")
        return components
        
    except Exception as e:
        logging.error(f"구성 요소 초기화 오류: {e}")
        raise ConfigurationError(f"초기화 실패: {str(e)}")

# 리스크 대시보드 업데이트 함수 (새로 추가)
async def update_risk_dashboard(components):
    """전체 시스템에서 리스크 대시보드 업데이트"""
    try:
        # 각 구성 요소에서 리스크 메트릭스 수집
        sources = {}
        
        # 시장 분석기에서 데이터 가져오기
        if 'market_analyzer' in components:
            market_analyzer = components['market_analyzer']
            analysis = await market_analyzer.get_integrated_analysis()
            if analysis and 'risk_metrics' in analysis:
                sources['market_analyzer'] = analysis['risk_metrics']
        
        # 리스크 관리자에서 데이터 가져오기
        if 'risk_manager' in components:
            risk_manager = components['risk_manager']
            risk_status = risk_manager.get_risk_status()
            if 'risk_metrics' in risk_status:
                sources['risk_manager'] = risk_status['risk_metrics']
            if 'var_levels' in risk_status:
                sources['risk_manager_var'] = risk_status['var_levels']
        
        # 뉴스 분석기에서 데이터 가져오기 (있을 경우)
        if 'news_analyzer' in components:
            news_analyzer = components['news_analyzer']
            news_stats = news_analyzer.get_analysis_stats()
            if news_stats:
                sources['news_analyzer'] = news_stats
        
        # 온체인 분석기 통합 (있을 경우)
        if 'blockchain_integration' in components:
            blockchain = components['blockchain_integration']
            network_state = await blockchain.get_network_state()
            if network_state:
                sources['onchain'] = network_state
        
        # 리스크 대시보드 업데이트
        risk_dashboard.update_metrics(sources)
        
        # 통합 리스크 메트릭스 로깅
        consolidated = risk_dashboard.get_consolidated_metrics()
        logging.info(f"통합 리스크 상태: 레벨={consolidated.get('risk_level', 'unknown')}, "
                     f"점수={consolidated.get('risk_score', 0):.2f}")
        
    except Exception as e:
        logging.error(f"리스크 대시보드 업데이트 오류: {e}")

# 메인 트레이딩 루프
async def trading_loop(components):
    """메인 트레이딩 루프 (리스크 대시보드 업데이트 추가됨)"""
    try:
        data_orchestrator = components['data_orchestrator']
        market_analyzer = components['market_analyzer']
        trading_system = components['trading_system']
        risk_manager = components['risk_manager']
        performance_monitor = components['performance_monitor']
        blockchain_integration = components.get('blockchain_integration')
        
        # 리스크 대시보드 초기 업데이트
        await update_risk_dashboard(components)
        
        dashboard_update_counter = 0  # 업데이트 주기 카운터
        
        while not shutdown_requested:
            try:
                # 데이터 수집
                start_time = time.time()
                collected_data = await data_orchestrator.collect_all()
                
                # 시장 분석
                analysis_result = await market_analyzer.get_integrated_analysis()
                
                # 리스크 체크
                risk_status = risk_manager.get_risk_status()
                
                # 온체인 데이터 통합 (새로 추가)
                onchain_data = {}
                if blockchain_integration:
                    onchain_data = await blockchain_integration.get_latest_scenarios(3)
                
                # 트레이딩 결정
                market_data = {
                    'price': collected_data.get('price_data', {}).get('price', 0),
                    'timestamp': datetime.now(),
                    'integrated_state': analysis_result,
                    'risk_status': risk_status,
                    'onchain_data': onchain_data  # 온체인 데이터 추가
                }
                
                # 트레이딩 신호 생성
                trading_signal = await trading_system._generate_trading_signal(market_data)
                
                # 신호가 있으면 실행
                if trading_signal and trading_signal.action != 'hold':
                    # 리스크 체크
                    trade_check = await risk_manager.check_trade(
                        {'symbol': trading_signal.symbol, 'size': trading_signal.size}
                    )
                    
                    if trade_check['is_safe']:
                        # 거래 실행
                        execution_result = await trading_system._execute_trade(trading_signal)
                        
                        # 성능 기록
                        performance_monitor.record_trade(
                            execution_result.success,
                            execution_result.pnl or 0,
                            time.time() - start_time
                        )
                
                # 10회마다 리스크 대시보드 업데이트 (매우 빈번한 업데이트 방지)
                dashboard_update_counter += 1
                if dashboard_update_counter >= 10:
                    await update_risk_dashboard(components)
                    dashboard_update_counter = 0
                
                # 실행 간격 조정 (밀리초 단위)
                elapsed = time.time() - start_time
                wait_time = max(0.001, Config.TRADE_INTERVAL - elapsed)
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logging.error(f"트레이딩 루프 오류: {e}")
                await asyncio.sleep(1)
        
        logging.info("트레이딩 루프 종료")
        
    except Exception as e:
        logging.error(f"트레이딩 프로세스 오류: {e}")
        raise

# 정상 종료 처리
async def shutdown_system(components):
    """시스템 정상 종료"""
    try:
        logging.info("시스템 종료 처리 시작...")
        
        # 구성 요소별 종료 처리
        for name, component in components.items():
            try:
                # stop 메서드가 있는 컴포넌트 확인
                if hasattr(component, 'stop') and callable(getattr(component, 'stop')):
                    await component.stop()
                elif hasattr(component, 'stop_collection') and callable(getattr(component, 'stop_collection')):
                    await component.stop_collection()
                elif hasattr(component, 'stop_monitoring') and callable(getattr(component, 'stop_monitoring')):
                    await component.stop_monitoring()
                elif hasattr(component, 'stop_integration') and callable(getattr(component, 'stop_integration')):
                    await component.stop_integration()
                    
                logging.info(f"{name} 정상 종료 완료")
                
            except Exception as e:
                logging.error(f"{name} 종료 오류: {e}")
        
        logging.info("모든 구성 요소 종료 완료")
        
    except Exception as e:
        logging.error(f"시스템 종료 오류: {e}")

# 메인 함수
async def main():
    """메인 함수"""
    global shutdown_requested
    shutdown_requested = False
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # 로깅 설정
    setup_logging()
    
    # 환경 변수 로드
    load_dotenv()
    
    logging.info("==== 프로젝트 인피니티 코인 v2 시작 ====")
    
    try:
        # 구성 요소 초기화
        components = await initialize_components()
        
        # 메인 트레이딩 루프 실행
        trading_task = asyncio.create_task(trading_loop(components))
        
        # 종료 신호 대기
        while not shutdown_requested:
            await asyncio.sleep(1)
        
        # 트레이딩 태스크 취소
        trading_task.cancel()
        try:
            await trading_task
        except asyncio.CancelledError:
            pass
        
        # 시스템 정상 종료
        await shutdown_system(components)
        
    except ConfigurationError as e:
        logging.critical(f"설정 오류: {e}")
    except SystemEmergencyError as e:
        logging.critical(f"시스템 긴급 상황: {e}")
    except Exception as e:
        logging.critical(f"예상치 못한 오류: {e}")
    finally:
        logging.info("==== 프로젝트 인피니티 코인 v2 종료 ====")

# 스크립트 직접 실행 시
if __name__ == "__main__":    
    # 비동기 이벤트 루프 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("사용자에 의한 종료")
    except Exception as e:
        logging.critical(f"치명적 오류: {e}")