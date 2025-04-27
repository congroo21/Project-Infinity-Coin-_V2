# src/scenarios/models/blockchain_integration.py

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ...analyzers.onchain_analyzer import OnchainAnalyzer
from ...analyzers.transaction_analyzer import TransactionAnalyzer
from ...collectors.onchain_collector import OnchainCollector, BlockchainConfig
from ...database import DatabaseManager
from ...models.onchain_models import MarketSignal, WhaleMovement
from ...exceptions import ValidationError, InsufficientDataError, MarketDataError

from ..models.base_models import MarketScenario
from ...market_analyzer import MarketState

@dataclass
class BlockchainIntegrationConfig:
    """블록체인 통합 설정"""
    signal_confidence_threshold: float = 0.6  # 신호 신뢰도 임계값
    integration_interval: float = 60.0  # 통합 주기 (초)
    whale_impact_threshold: float = 0.7  # 웨일 영향도 임계값
    network_congestion_threshold: float = 0.7  # 네트워크 혼잡도 임계값
    enable_real_time_signals: bool = True  # 실시간 신호 활성화
    enable_historical_analysis: bool = True  # 과거 데이터 분석 활성화

class BlockchainIntegration:
    """블록체인 데이터 통합 모듈"""
    def __init__(
        self, 
        onchain_analyzer: OnchainAnalyzer, 
        transaction_analyzer: TransactionAnalyzer,
        config: Optional[BlockchainIntegrationConfig] = None
    ):
        self.onchain_analyzer = onchain_analyzer
        self.transaction_analyzer = transaction_analyzer
        self.config = config or BlockchainIntegrationConfig()
        
        # 통합 결과 저장소
        self.integrated_scenarios = []
        self.last_integration = None
        
        # 실행 상태
        self.running = False
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
    
    async def start_integration(self):
        """통합 프로세스 시작"""
        self.running = True
        self.logger.info("Starting blockchain integration process")
        
        while self.running:
            try:
                # 온체인 신호 수집
                onchain_signals = await self.onchain_analyzer.get_latest_signals()
                
                if not onchain_signals:
                    self.logger.info("No onchain signals available")
                    await asyncio.sleep(self.config.integration_interval)
                    continue
                
                # 웨일 활동 분석
                whale_activity = await self.onchain_analyzer.get_whale_activity()
                
                # 네트워크 상태 조회
                network_state = await self.onchain_analyzer.get_network_state()
                
                # 가스 가격 추세 분석
                gas_trend = await self.transaction_analyzer.get_gas_price_trend()
                
                # 시장 영향 예측
                market_impact = await self.onchain_analyzer.get_market_impact_prediction()
                
                # 통합 시나리오 생성
                scenarios = await self._create_integrated_scenarios(
                    onchain_signals, whale_activity, network_state, 
                    gas_trend, market_impact
                )
                
                # 결과 저장
                if scenarios:
                    self.integrated_scenarios.extend(scenarios)
                    self.logger.info(f"Created {len(scenarios)} integrated scenarios")
                
                self.last_integration = datetime.now()
                
                # 다음 통합까지 대기
                await asyncio.sleep(self.config.integration_interval)
                
            except Exception as e:
                self.logger.error(f"Integration process error: {e}")
                await asyncio.sleep(5)  # 에러 발생 시 5초 대기
    
    async def stop_integration(self):
        """통합 프로세스 중지"""
        self.running = False
        self.logger.info("Stopping blockchain integration process")
    
    async def _create_integrated_scenarios(
        self,
        onchain_signals: List[Dict],
        whale_activity: Dict,
        network_state: Dict,
        gas_trend: Dict,
        market_impact: Dict
    ) -> List[MarketScenario]:
        """통합 시나리오 생성"""
        try:
            # 입력 데이터 검증
            if onchain_signals is None:
                onchain_signals = []
            if whale_activity is None:
                whale_activity = {}
            if network_state is None:
                network_state = {}
            if gas_trend is None:
                gas_trend = {}
            if market_impact is None:
                market_impact = {}
                
            self.logger.info("통합 시나리오 생성 시작")
            
            scenarios = []
            
            # 기본 시나리오 (항상 하나 이상의 시나리오 보장)
            default_scenario = MarketScenario(
                timestamp=datetime.now(),
                scenario_type='neutral_market',
                probability=0.5,
                risk_score=0.5,
                expected_return=0.0,
                suggested_position='neutral',
                confidence_score=0.5,
                parameters={
                    'source': 'default',
                    'note': '데이터 부족으로 인한 기본 시나리오'
                }
            )
            scenarios.append(default_scenario)
            
            # 데이터가 있을 경우 시나리오 생성
            # 웨일 움직임 시나리오
            if whale_activity.get('movements_count', 0) > 0:
                whale_scenario = MarketScenario(
                    timestamp=datetime.now(),
                    scenario_type='whale_movement',
                    probability=0.7,
                    risk_score=0.6,
                    expected_return=0.01,
                    suggested_position='neutral',
                    confidence_score=0.6,
                    parameters={
                        'whale_movements': whale_activity.get('movements_count', 0),
                        'source': 'whale_activity'
                    }
                )
                scenarios.append(whale_scenario)
                
            # 네트워크 상태 시나리오
            if network_state and network_state.get('average_gas_price', 0) > 0:
                network_scenario = MarketScenario(
                    timestamp=datetime.now(),
                    scenario_type='network_state',
                    probability=0.6,
                    risk_score=0.5,
                    expected_return=0.005,
                    suggested_position='neutral',
                    confidence_score=0.6,
                    parameters={
                        'gas_price': network_state.get('average_gas_price', 0),
                        'source': 'network_state'
                    }
                )
                scenarios.append(network_scenario)
                
            self.logger.info(f"생성된 시나리오: {len(scenarios)}개")
            return scenarios
            
        except Exception as e:
            self.logger.error(f"Scenario creation error: {e}")
            # 기본 시나리오 반환
            return [
                MarketScenario(
                    timestamp=datetime.now(),
                    scenario_type='error_fallback',
                    probability=0.5,
                    risk_score=0.5,
                    expected_return=0.0,
                    suggested_position='neutral',
                    confidence_score=0.5,
                    parameters={
                        'error': str(e),
                        'source': 'error_handler'
                    }
                )
            ]
    
    async def get_latest_scenarios(self, limit: int = 5) -> List[Dict]:
        """최신 통합 시나리오 조회"""
        try:
            scenarios = self.integrated_scenarios[-limit:] if self.integrated_scenarios else []
            
            result = []
            for scenario in scenarios:
                scenario_dict = {
                    'timestamp': scenario.timestamp.isoformat(),
                    'type': scenario.scenario_type,
                    'probability': scenario.probability,
                    'risk_score': scenario.risk_score,
                    'expected_return': scenario.expected_return,
                    'suggested_position': scenario.suggested_position,
                    'confidence': scenario.confidence_score,
                    'parameters': scenario.parameters
                }
                result.append(scenario_dict)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting latest scenarios: {e}")
            return []
    
    async def integrate_with_market_state(self, market_state: MarketState) -> Dict:
        """온체인 데이터와 시장 상태 통합"""
        try:
            if not self.integrated_scenarios:
                return {'integrated': False}
            
            # 최신 시나리오
            latest_scenarios = self.integrated_scenarios[-5:]
            
            # 온체인 신호
            onchain_signals = await self.onchain_analyzer.get_latest_signals(5)
            
            # 가중치 설정
            weights = {
                'whale_accumulation': 0.3,
                'whale_distribution': 0.3,
                'network_congestion': 0.1,
                'rising_gas_prices': 0.1,
                'falling_gas_prices': 0.1,
                'onchain_bullish': 0.2,
                'onchain_bearish': 0.2
            }
            
            # 최종 시그널 계산
            final_signal = 0
            total_weight = 0
            
            for scenario in latest_scenarios:
                scenario_type = scenario.scenario_type
                weight = weights.get(scenario_type, 0.1)
                
                # 시그널 방향 (-1: bearish, 0: neutral, 1: bullish)
                direction = 0
                if 'bullish' in scenario_type or 'accumulation' in scenario_type:
                    direction = 1
                elif 'bearish' in scenario_type or 'distribution' in scenario_type:
                    direction = -1
                
                # 가중치 적용
                final_signal += direction * weight * scenario.confidence_score
                total_weight += weight
            
            # 정규화
            if total_weight > 0:
                final_signal /= total_weight
            
            # 시장 상태와 통합
            integrated_state = {
                'timestamp': datetime.now().isoformat(),
                'market_trend': market_state.trend,
                'onchain_signal': 'bullish' if final_signal > 0.2 else 'bearish' if final_signal < -0.2 else 'neutral',
                'signal_strength': abs(final_signal),
                'suggested_action': 'buy' if final_signal > 0.3 else 'sell' if final_signal < -0.3 else 'hold',
                'confidence': min(1.0, 0.5 + abs(final_signal)),
                'supporting_scenarios': len(latest_scenarios),
                'onchain_signals': len(onchain_signals)
            }
            
            return integrated_state
            
        except Exception as e:
            self.logger.error(f"Market state integration error: {e}")
            return {'integrated': False, 'error': str(e)}

# 블록체인 통합 시스템 초기화 및 실행 헬퍼 함수
async def initialize_blockchain_integration(
    db_manager: DatabaseManager,
    rpc_url: str,
    chain_id: int = 1
) -> BlockchainIntegration:
    """블록체인 통합 시스템 초기화"""
    try:
        # 블록체인 설정
        blockchain_config = BlockchainConfig(
            rpc_url=rpc_url,
            chain_id=chain_id,
            update_interval=10.0
        )
        
        # 온체인 데이터 수집기
        collector = OnchainCollector(blockchain_config, db_manager)
        await collector.initialize()
        
        # 온체인 분석기
        analyzer = OnchainAnalyzer(collector)
        await analyzer.initialize()
        
        # 트랜잭션 분석기
        tx_analyzer = TransactionAnalyzer(collector)
        
        # 블록체인 통합 모듈
        integration = BlockchainIntegration(analyzer, tx_analyzer)
        
        return integration
        
    except Exception as e:
        logging.error(f"Blockchain integration initialization error: {e}")
        raise