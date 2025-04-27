# src/analyzers/onchain_analyzer.py

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque, Counter
import numpy as np
from decimal import Decimal
from dataclasses import dataclass, field

from ..collectors.onchain_collector import OnchainCollector
from ..models.onchain_models import (
    BlockData, TransactionData, TokenTransferData, 
    BlockchainMetrics, WhaleMovement, MarketSignal,
    NetworkState
)
from ..exceptions import ValidationError, InsufficientDataError, MarketDataError

@dataclass
class OnchainAnalysisConfig:
    """온체인 분석 설정"""
    whale_threshold_eth: float = 100.0  # 웨일 트랜잭션 최소 ETH 값
    significant_token_moves_usd: float = 50000.0  # 중요 토큰 이동 최소 USD 값
    token_price_refresh_interval: int = 300  # 토큰 가격 갱신 주기 (초)
    lookback_blocks: int = 1000  # 분석 대상 블록 수
    gas_price_impact_threshold: float = 20.0  # 가스 가격 영향 임계값 (Gwei)
    analysis_interval: float = 60.0  # 분석 주기 (초)
    exchange_addresses: List[str] = field(default_factory=list)  # 거래소 주소 목록

class OnchainAnalyzer:
    """온체인 데이터 분석기"""
    def __init__(self, onchain_collector: OnchainCollector, config: Optional[OnchainAnalysisConfig] = None):
        self.collector = onchain_collector
        self.config = config or OnchainAnalysisConfig()
        
        # 분석 결과 캐시
        self.metrics_cache = deque(maxlen=1000)
        self.whale_movements_cache = deque(maxlen=100)
        self.network_state_cache = deque(maxlen=100)
        self.signals_cache = deque(maxlen=500)
        
        # 토큰 가격 캐시
        self.token_prices = {}
        self.last_price_update = None
        
        # 주소 카테고리 (거래소, 유명 지갑 등)
        self.address_categories = {}
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        # 분석 상태
        self.running = False
        self.last_analysis = None
    
    async def initialize(self):
        """분석기 초기화"""
        try:
            # 중요 주소 카테고리 로딩
            await self._load_address_categories()
            
            # 토큰 가격 초기화
            await self._update_token_prices()
            
            # 초기 네트워크 상태 분석
            await self._analyze_network_state()
            
            self.logger.info("Onchain analyzer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            return False
    
    async def start_analysis(self):
        """분석 루프 시작"""
        self.running = True
        self.logger.info("Starting onchain analysis loop")
        
        while self.running:
            try:
                # 네트워크 상태 분석
                network_state = await self._analyze_network_state()
                
                # 웨일 움직임 분석
                whale_movements = await self._analyze_whale_movements()
                
                # 토큰 흐름 분석
                token_flows = await self._analyze_token_flows()
                
                # 신호 생성
                signals = await self._generate_market_signals(
                    network_state, whale_movements, token_flows
                )
                
                # 결과 저장
                if signals:
                    for signal in signals:
                        self.signals_cache.append(signal)
                
                self.last_analysis = datetime.now()
                
                # 다음 분석까지 대기
                await asyncio.sleep(self.config.analysis_interval)
                
            except Exception as e:
                self.logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(5)  # 에러 발생 시 5초 대기
    
    async def stop_analysis(self):
        """분석 루프 중지"""
        self.running = False
        self.logger.info("Stopping onchain analysis loop")
    
    async def _load_address_categories(self):
        """주요 주소 카테고리 로딩 (거래소, 지갑 등)"""
        try:
            # 여기서는 간단한 예시로 하드코딩
            # 실제로는 DB나 외부 API에서 로딩해야 함
            self.address_categories = {
                # 대표적인 중앙화 거래소 주소 (예시)
                "0x28c6c06298d514db089934071355e5743bf21d60": "binance",
                "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "binance",
                "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "binance",
                "0x5a52e96bacdabb82fd05763e25335261b270efcb": "ftx",
                "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": "binance",
                
                # 대형 유동성 풀/프로토콜 (예시)
                "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "uniswap_router",
                "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "uniswap_router_v3",
                "0x1111111254fb6c44bac0bed2854e76f90643097d": "1inch_router",
                
                # 스테이블코인 발행사 (예시)
                "0xc6cde7c39eb2f0f0095f41570af89efc2c1ea828": "usdc_treasury",
                "0x5754284f345afc66a98fbb0a0afe71e0f007b949": "tether_treasury"
            }
            
            # 설정에서 거래소 주소 추가
            for address in self.config.exchange_addresses:
                if address not in self.address_categories:
                    self.address_categories[address.lower()] = "exchange"
            
            self.logger.info(f"Loaded {len(self.address_categories)} address categories")
            
        except Exception as e:
            self.logger.error(f"Error loading address categories: {e}")
            # 기본 카테고리 초기화
            self.address_categories = {}
    
    async def _update_token_prices(self):
        """토큰 가격 갱신"""
        try:
            # 마지막 갱신 시간 확인
            now = datetime.now()
            if (self.last_price_update and 
                (now - self.last_price_update).total_seconds() < self.config.token_price_refresh_interval):
                return
            
            # 여기에서 가격 API를 호출하여 주요 토큰 가격 갱신
            # 예시 코드 (실제로는 CoinGecko, CoinMarketCap 등의 API 사용)
            self.token_prices = {
                "ETH": 2500.0,
                "BTC": 40000.0,
                "USDT": 1.0,
                "USDC": 1.0,
                "DAI": 1.0,
                # 다른 토큰 가격 추가
            }
            
            self.last_price_update = now
            self.logger.info("Token prices updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating token prices: {e}")
    
    async def _analyze_network_state(self) -> Optional[NetworkState]:
        """블록체인 네트워크 상태 분석"""
        try:
            # 온체인 메트릭스 수집
            metrics = await self.collector.get_onchain_metrics()
            if not metrics:
                # 기본 메트릭스 생성
                self.logger.warning("온체인 메트릭스를 가져올 수 없습니다. 기본값 사용.")
                metrics = {
                    'block_time_seconds': 13.0,  # 이더리움 기본 블록 시간
                    'gas_price_gwei': 20.0,      # 기본 가스 가격
                    'mempool_size': 0,
                    'avg_transactions_per_block': 100
                }
            
            # 블록체인 메트릭스 저장
            self.metrics_cache.append(metrics)
            
            # 네트워크 혼잡도 계산
            block_time = metrics.get('block_time_seconds', 13.0)  # 기본값 사용
            gas_price = metrics.get('gas_price_gwei', 20.0)      # 기본값 사용
            mempool_size = metrics.get('mempool_size', 0)
            
            # 데이터 유효성 검증 및 변환
            if block_time is None or block_time <= 0:
                block_time = 13.0  # 기본값 설정
                
            if gas_price is None or gas_price <= 0:
                gas_price = 20.0  # 기본값 설정
                
            # decimal.Decimal 타입과 float 연산 문제 해결을 위한 변환
            block_time_float = float(block_time) if isinstance(block_time, Decimal) else block_time
            congestion_block_time = min(1.0, block_time_float / 15.0)
            
            # 가스 가격 기준 혼잡도
            gas_price_float = float(gas_price) if isinstance(gas_price, Decimal) else gas_price
            congestion_gas = min(1.0, gas_price_float / 20.0)
            
            # 멤풀 크기 기준 혼잡도
            mempool_size_float = float(mempool_size) if isinstance(mempool_size, Decimal) else mempool_size
            congestion_mempool = min(1.0, mempool_size_float / 10000.0)
            
            # 종합 혼잡도 (가중 평균)
            congestion_score = (
                congestion_block_time * 0.3 +
                congestion_gas * 0.5 +
                congestion_mempool * 0.2
            )
            
            # 활성 주소 수 (예시 값)
            active_addresses = 10000
            tx_count_24h = 1000000
            
            # 네트워크 상태 객체 생성
            state = NetworkState(
                timestamp=datetime.now(),
                chain_id=self.collector.blockchain_config.chain_id,
                block_height=self.collector.last_block_number,
                is_congested=congestion_score > 0.7,
                average_block_time=block_time_float,
                average_gas_price=gas_price_float,
                active_addresses_24h=active_addresses,
                transaction_count_24h=tx_count_24h
            )
            
            self.logger.info(f"네트워크 상태 분석 완료: 혼잡도={congestion_score:.2f}, 가스={gas_price_float:.2f} Gwei")
            
            # 캐시에 저장
            self.network_state_cache.append(state)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Network state analysis error: {e}")
            # 기본 네트워크 상태 반환
            return NetworkState(
                timestamp=datetime.now(),
                chain_id=self.collector.blockchain_config.chain_id,
                block_height=self.collector.last_block_number,
                is_congested=False,
                average_block_time=13.0,
                average_gas_price=20.0,
                active_addresses_24h=0,
                transaction_count_24h=0
            )
    
    async def _analyze_whale_movements(self) -> List[WhaleMovement]:
        """웨일 움직임 분석 - 개선된 버전"""
        try:
            # 대규모 거래 조회 - 임계값을 낮춤 (100 ETH → 10 ETH)
            movements = await self.collector.get_whale_movements(
                min_value_eth=10.0  # 임계값 낮춤 (기존 self.config.whale_threshold_eth에서 변경)
            )
            
            self.logger.info(f"대규모 거래 감지: {len(movements)}개")
            
            if not movements:
                return []
            
            whale_movements = []
            for movement in movements:
                # 주소 카테고리 확인
                from_address = movement.get('from', '').lower()
                to_address = movement.get('to', '').lower()
                
                # 기존 카테고리 조회
                from_category = self._get_address_category(from_address)
                to_category = self._get_address_category(to_address)
                
                # 주소 카테고리가 'unknown'인 경우 추가 분석 시도
                if from_category == 'unknown':
                    # 대규모 자금 출금이면 웨일로 간주
                    if movement.get('value', 0) > 20.0:  # 20 ETH 이상
                        from_category = 'whale'
                
                if to_category == 'unknown':
                    # 대규모 자금 입금이면 웨일로 간주
                    if movement.get('value', 0) > 20.0:  # 20 ETH 이상
                        to_category = 'whale'
                
                # 동적 주소 카테고리 업데이트 (나중에 참조할 수 있도록)
                if from_category == 'whale':
                    self.address_categories[from_address] = 'whale'
                if to_category == 'whale':
                    self.address_categories[to_address] = 'whale'
                
                # 거래 유형 분류
                if from_category == 'exchange' and to_category != 'exchange':
                    category = 'exchange_withdrawal'
                elif from_category != 'exchange' and to_category == 'exchange':
                    category = 'exchange_deposit'
                elif from_category == 'whale' or to_category == 'whale':
                    category = 'whale_transfer'  # 추가된 카테고리
                else:
                    category = 'wallet_transfer'
                
                # 영향도 점수 계산
                impact_score = self._calculate_movement_impact(
                    movement.get('value', 0), category, from_category, to_category
                )
                
                # 블록 타임스탬프가 있으면 사용, 없으면 현재 시간
                timestamp = movement.get('timestamp', datetime.now())
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except:
                        timestamp = datetime.now()
                
                # WhaleMovement 객체 생성
                whale_movement = WhaleMovement(
                    tx_hash=movement.get('hash', ''),
                    block_number=movement.get('block_number', 0),
                    timestamp=timestamp,
                    from_address=from_address,
                    to_address=to_address,
                    value=movement.get('value', 0),
                    category=category,
                    impact_score=impact_score
                )
                
                # 로깅 추가
                self.logger.info(f"웨일 움직임 감지: {whale_movement.value} ETH, 유형: {category}, 영향도: {impact_score:.2f}")
                
                whale_movements.append(whale_movement)
            
            # 캐시에 저장
            for wm in whale_movements:
                self.whale_movements_cache.append(wm)
            
            self.logger.info(f"분석된 웨일 움직임: {len(whale_movements)}개")
            return whale_movements
            
        except Exception as e:
            self.logger.error(f"Whale movement analysis error: {e}")
            return []
    
    def _get_address_category(self, address: str) -> str:
        """주소 카테고리 조회"""
        address = address.lower() if address else ""
        return self.address_categories.get(address, "unknown")
    
    def _calculate_movement_impact(
        self, 
        value: float, 
        category: str, 
        from_category: str,
        to_category: str
    ) -> float:
        """웨일 움직임의 시장 영향도 계산 - 개선된 버전"""
        try:
            # 기본 영향도 (금액에 따른 스케일링) - 더 민감하게 조정
            base_impact = min(1.0, value / 500.0)  # 500 ETH → 1.0 (기존 1000 ETH에서 변경)
            
            # 카테고리별 가중치 조정 - 더 민감하게 조정
            category_multiplier = {
                'exchange_withdrawal': 1.5,   # 거래소 출금은 매수 압력 시사 (기존 1.2에서 상향)
                'exchange_deposit': 1.2,      # 거래소 입금은 매도 압력 시사 (기존 0.8에서 상향)
                'whale_transfer': 1.3,        # 웨일 간 이체 (새로 추가)
                'wallet_transfer': 0.8        # 일반 지갑 이체 (기존 0.5에서 상향)
            }.get(category, 1.0)
            
            # 최종 영향도 계산
            impact = base_impact * category_multiplier
            
            return min(1.0, impact)
            
        except Exception as e:
            self.logger.error(f"Impact calculation error: {e}")
            return 0.5  # 기본값
    
    async def _analyze_token_flows(self) -> Dict:
        """토큰 흐름 분석"""
        # 예시 구현
        return {
            'net_exchange_flow': 10.5,  # 예시: 순 거래소 흐름 (ETH)
            'largest_token_transfer': {
                'token': 'USDT',
                'amount': 1000000.0,
                'from_type': 'whale',
                'to_type': 'exchange'
            }
        }
    
    async def _generate_market_signals(
        self,
        network_state: Optional[NetworkState],
        whale_movements: List[WhaleMovement],
        token_flows: Dict
    ) -> List[MarketSignal]:
        """시장 신호 생성 - 개선된 버전"""
        try:
            signals = []
            
            # 1. 네트워크 상태 기반 신호 - 변경 없음
            if network_state:
                if network_state.is_congested:
                    # 네트워크 혼잡은 높은 활동성을 의미하며, 가격 변동성 증가 가능성
                    congestion_signal = MarketSignal(
                        timestamp=datetime.now(),
                        signal_type='volatile',
                        strength=min(1.0, float(network_state.average_gas_price) / 30.0),
                        time_horizon='short',
                        confidence=0.7,
                        source='network_congestion',
                        related_assets=['ETH'],
                        supporting_data={
                            'gas_price_gwei': float(network_state.average_gas_price),
                            'block_time': float(network_state.average_block_time)
                        },
                        predicted_impact={'ETH': 0.02}
                    )
                    signals.append(congestion_signal)
            
            # 2. 웨일 움직임 기반 신호 - 더 민감하게 조정
            if whale_movements:
                # 거래소 입출금 패턴 분석
                deposits = [wm for wm in whale_movements if wm.category == 'exchange_deposit']
                withdrawals = [wm for wm in whale_movements if wm.category == 'exchange_withdrawal']
                
                deposit_volume = sum(wm.value for wm in deposits)
                withdrawal_volume = sum(wm.value for wm in withdrawals)
                
                # 순 흐름 계산
                net_flow = withdrawal_volume - deposit_volume
                
                # 임계값을 낮춤 (100 ETH → 20 ETH)
                if abs(net_flow) > 20:  # 20 ETH 이상의 순 흐름이 있을 때
                    flow_signal_type = 'bullish' if net_flow > 0 else 'bearish'
                    flow_strength = min(1.0, abs(net_flow) / 200.0)  # 200 ETH가 최대 강도 (기존 500 ETH에서 변경)
                    
                    flow_signal = MarketSignal(
                        timestamp=datetime.now(),
                        signal_type=flow_signal_type,
                        strength=flow_strength,
                        time_horizon='medium',
                        confidence=0.65,
                        source='whale_exchange_flow',
                        related_assets=['ETH'],
                        supporting_data={
                            'net_flow_eth': net_flow,
                            'deposit_count': len(deposits),
                            'withdrawal_count': len(withdrawals)
                        },
                        predicted_impact={'ETH': 0.03 if flow_signal_type == 'bullish' else -0.03}
                    )
                    signals.append(flow_signal)
                
                # 개별 웨일 거래에 대한 신호도 추가 (신규 추가)
                for wm in whale_movements:
                    if wm.impact_score > 0.3:  # 중요도 높은 웨일 움직임만
                        movement_type = ''
                        if wm.category == 'exchange_withdrawal':
                            movement_type = 'bullish'
                        elif wm.category == 'exchange_deposit':
                            movement_type = 'bearish'
                        else:
                            continue  # 중요하지 않은 움직임은 무시
                        
                        whale_signal = MarketSignal(
                            timestamp=datetime.now(),
                            signal_type=movement_type,
                            strength=wm.impact_score,
                            time_horizon='short',
                            confidence=0.6,
                            source='individual_whale_movement',
                            related_assets=['ETH'],
                            supporting_data={
                                'transaction_hash': wm.tx_hash,
                                'value_eth': wm.value,
                                'category': wm.category
                            },
                            predicted_impact={'ETH': 0.01 if movement_type == 'bullish' else -0.01}
                        )
                        signals.append(whale_signal)
            
            # 3. 토큰 흐름 기반 신호 - 더 민감하게 조정
            if token_flows.get('net_exchange_flow'):
                net_exchange_flow = token_flows['net_exchange_flow']
                # 임계값 낮춤 (50 ETH → 10 ETH)
                if abs(net_exchange_flow) > 10:  # 10 ETH 이상의 순 흐름
                    token_signal_type = 'bullish' if net_exchange_flow > 0 else 'bearish'
                    
                    token_signal = MarketSignal(
                        timestamp=datetime.now(),
                        signal_type=token_signal_type,
                        strength=min(1.0, abs(net_exchange_flow) / 100.0),  # 100 ETH가 최대 강도 (기존 200 ETH에서 변경)
                        time_horizon='short',
                        confidence=0.6,
                        source='token_flow',
                        related_assets=['ETH'],
                        supporting_data={
                            'net_exchange_flow': net_exchange_flow
                        },
                        predicted_impact={'ETH': 0.02 if token_signal_type == 'bullish' else -0.02}
                    )
                    signals.append(token_signal)
            
            self.logger.info(f"생성된 시장 신호: {len(signals)}개")
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return []
    
    async def get_latest_signals(self, limit: int = 10) -> List[Dict]:
        """최신 시장 신호 조회"""
        try:
            signals = list(self.signals_cache)[-limit:]
            return [signal.to_dict() for signal in signals]
        except Exception as e:
            self.logger.error(f"Error getting latest signals: {e}")
            return []
    
    async def get_network_state(self) -> Dict:
        """현재 네트워크 상태 조회"""
        try:
            if not self.network_state_cache:
                return {}
                
            latest_state = self.network_state_cache[-1]
            return latest_state.to_dict()
        except Exception as e:
            self.logger.error(f"Error getting network state: {e}")
            return {}
    
    async def get_whale_activity(self, hours: int = 24) -> Dict:
        """웨일 활동 통계"""
        try:
            # 지정된 시간 내의 웨일 움직임 필터링
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_movements = [
                wm for wm in self.whale_movements_cache
                if wm.timestamp > cutoff_time
            ]
            
            if not recent_movements:
                return {'movements_count': 0}
            
            # 카테고리별 집계
            categories = Counter([wm.category for wm in recent_movements])
            
            # 총 거래량
            total_volume = sum(wm.value for wm in recent_movements)
            
            # 순 거래소 흐름
            deposits = sum(wm.value for wm in recent_movements if wm.category == 'exchange_deposit')
            withdrawals = sum(wm.value for wm in recent_movements if wm.category == 'exchange_withdrawal')
            net_exchange_flow = withdrawals - deposits
            
            return {
                'movements_count': len(recent_movements),
                'categories': dict(categories),
                'total_volume_eth': total_volume,
                'net_exchange_flow_eth': net_exchange_flow,
                'bullish_indicator': 1 if net_exchange_flow > 0 else -1 if net_exchange_flow < 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting whale activity: {e}")
            return {}
    
    async def get_market_impact_prediction(self) -> Dict:
        """온체인 데이터 기반 시장 영향 예측"""
        try:
            signals = list(self.signals_cache)[-20:]  # 최근 20개 신호
            
            if not signals:
                return {'impact_prediction': 'neutral', 'confidence': 0.0}
            
            # 가중치 계산 (신호 강도와 신뢰도 반영)
            weights = [signal.strength * signal.confidence for signal in signals]
            
            # 시그널 유형별 점수 (bullish: 1, bearish: -1, neutral/volatile: 0)
            scores = []
            for signal in signals:
                if signal.signal_type == 'bullish':
                    scores.append(1)
                elif signal.signal_type == 'bearish':
                    scores.append(-1)
                else:
                    scores.append(0)
            
            # 가중 평균 계산
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights) if sum(weights) > 0 else 0
            
            # 영향 예측
            impact_prediction = 'neutral'
            if weighted_score > 0.3:
                impact_prediction = 'bullish'
            elif weighted_score < -0.3:
                impact_prediction = 'bearish'
            
            # 신뢰도 계산
            confidence = abs(weighted_score) * 0.7 + 0.3  # 0.3 ~ 1.0 범위
            
            return {
                'impact_prediction': impact_prediction,
                'weighted_score': weighted_score,
                'confidence': confidence,
                'time_horizon': 'short',
                'supporting_signals_count': len(signals)
            }
            
        except Exception as e:
            self.logger.error(f"Impact prediction error: {e}")
            return {'impact_prediction': 'neutral', 'confidence': 0.0}