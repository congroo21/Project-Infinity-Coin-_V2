# src/analyzers/transaction_analyzer.py

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque, Counter
import numpy as np
from web3 import Web3

from ..collectors.onchain_collector import OnchainCollector
from ..models.onchain_models import (
    TransactionData, TokenTransferData, WhaleMovement
)

class TransactionAnalyzer:
    """블록체인 트랜잭션 분석기"""
    def __init__(self, onchain_collector: OnchainCollector):
        self.collector = onchain_collector
        
        # 분석 결과 캐시
        self.analyzed_txs = deque(maxlen=1000)
        self.gas_price_history = deque(maxlen=1000)
        self.whale_transfers = deque(maxlen=100)
        
        # 주소 패턴 분석
        self.address_activity = {}
        self.address_patterns = {}
        
        # 유명 컨트랙트 주소 (예시)
        self.known_contracts = {
            # DEX 라우터
            "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "uniswap_v2_router",
            "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45": "uniswap_v3_router",
            
            # 대형 유동성 풀
            "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc": "usdc_eth_pool",
            
            # 스테이블코인
            "0xdac17f958d2ee523a2206206994597c13d831ec7": "usdt",
            "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "usdc",
            
            # 유동성 스테이킹
            "0xc2edad668740f1aa35e4d8f227fb8e17dca888cd": "aave",
            
            # NFT 마켓플레이스
            "0x7be8076f4ea4a4ad08075c2508e481d6c946d12b": "opensea",
        }
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
    
    async def analyze_transaction(self, tx_hash: str) -> Dict:
        """트랜잭션 상세 분석"""
        try:
            # 로깅 추가
            self.logger.info(f"트랜잭션 분석 시작: {tx_hash}")
            
            # 트랜잭션 정보 조회
            tx_details = await self.collector.get_transaction_details(tx_hash)
            if not tx_details:
                # 기본 응답 제공
                self.logger.warning(f"Transaction {tx_hash} not found")
                return {
                    'tx_hash': tx_hash,
                    'type': 'unknown',
                    'contract_interaction': False,
                    'contract_type': 'unknown',
                    'value_eth': 0,
                    'gas_cost_eth': 0,
                    'timestamp': datetime.now().isoformat(),
                    'status': None,
                    'block_number': 0,
                    'error': 'Transaction not found'
                }
            
            # 컨트랙트 주소 확인
            to_address = tx_details.get('to', '')
            contract_type = self._identify_contract_type(to_address)
            
            # 거래 유형 분류
            tx_type = self._classify_transaction_type(tx_details, contract_type)
            
            # 가스 비용 계산
            gas_cost_eth = tx_details.get('gas_used', 0) * tx_details.get('gas_price', 0) / 1e18
            
            # 분석 결과
            analysis = {
                'tx_hash': tx_hash,
                'type': tx_type,
                'contract_interaction': contract_type != 'unknown',
                'contract_type': contract_type,
                'value_eth': tx_details.get('value', 0),
                'gas_cost_eth': gas_cost_eth,
                'timestamp': tx_details.get('timestamp'),
                'status': tx_details.get('status'),
                'block_number': tx_details.get('block_number')
            }
            
            # 주소 패턴 업데이트
            self._update_address_patterns(tx_details)
            
            # 분석 결과 캐시에 저장
            self.analyzed_txs.append(analysis)
            self.logger.info(f"트랜잭션 분석 완료: {tx_hash}, 유형: {tx_type}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Transaction analysis error: {e}")
            # 오류 발생시 기본 응답
            return {
                'tx_hash': tx_hash,
                'type': 'error',
                'error_message': str(e)
            }
    
    def _identify_contract_type(self, address: str) -> str:
        """컨트랙트 유형 식별"""
        if not address:
            return "contract_creation"
            
        address = address.lower()
        return self.known_contracts.get(address, "unknown")
    
    def _classify_transaction_type(self, tx_details: Dict, contract_type: str) -> str:
        """트랜잭션 유형 분류"""
        if not tx_details.get('to'):
            return "contract_creation"
            
        value = tx_details.get('value', 0)
        
        if contract_type == "unknown":
            return "ether_transfer" if value > 0 else "contract_interaction"
            
        # 컨트랙트 유형별 분류
        if "uniswap" in contract_type:
            return "dex_swap"
        elif "pool" in contract_type:
            return "liquidity_provision"
        elif contract_type in ["usdt", "usdc"]:
            return "stablecoin_transfer"
        elif contract_type == "aave":
            return "defi_staking"
        elif contract_type == "opensea":
            return "nft_transaction"
            
        return "contract_interaction"
    
    def _update_address_patterns(self, tx_details: Dict):
        """주소 활동 패턴 업데이트"""
        try:
            from_address = tx_details.get('from', '')
            to_address = tx_details.get('to', '')
            
            if not from_address:
                return
                
            # 발신자 주소 활동 기록
            if from_address not in self.address_activity:
                self.address_activity[from_address] = {
                    'tx_count': 0,
                    'total_eth_sent': 0,
                    'contract_interactions': 0,
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now()
                }
            
            activity = self.address_activity[from_address]
            activity['tx_count'] += 1
            activity['total_eth_sent'] += tx_details.get('value', 0)
            
            if to_address and to_address in self.known_contracts:
                activity['contract_interactions'] += 1
                
            activity['last_seen'] = tx_details.get('timestamp', datetime.now())
            
        except Exception as e:
            self.logger.error(f"Address pattern update error: {e}")
    
    async def analyze_token_transfers(self, token_address: str, hours: int = 24) -> Dict:
        """토큰 전송 분석"""
        try:
            # 최근 블록에서 시작 블록 계산 (예: 1시간에 240블록, 이더리움 기준)
            from_block = self.collector.last_block_number - (hours * 240)
            to_block = self.collector.last_block_number
            
            # 토큰 전송 이벤트 조회
            transfers = await self.collector.get_token_transfers(
                token_address, from_block, to_block
            )
            
            if not transfers:
                return {
                    'token_address': token_address,
                    'transfers_count': 0,
                    'timeframe_hours': hours
                }
            
            # 전송 분석
            total_tokens = sum(t['value'] for t in transfers)
            unique_senders = len(set(t['from'] for t in transfers))
            unique_receivers = len(set(t['to'] for t in transfers))
            
            # 주요 발신자/수신자 분석
            sender_count = Counter(t['from'] for t in transfers)
            receiver_count = Counter(t['to'] for t in transfers)
            
            top_senders = [(addr, count) for addr, count in sender_count.most_common(5)]
            top_receivers = [(addr, count) for addr, count in receiver_count.most_common(5)]
            
            # 대규모 전송 식별 (상위 10% 이상)
            values = [t['value'] for t in transfers]
            large_transfer_threshold = np.percentile(values, 90)
            large_transfers = [t for t in transfers if t['value'] >= large_transfer_threshold]
            
            return {
                'token_address': token_address,
                'transfers_count': len(transfers),
                'unique_senders': unique_senders,
                'unique_receivers': unique_receivers,
                'total_tokens_transferred': total_tokens,
                'top_senders': top_senders,
                'top_receivers': top_receivers,
                'large_transfers_count': len(large_transfers),
                'large_transfer_threshold': large_transfer_threshold,
                'timeframe_hours': hours
            }
            
        except Exception as e:
            self.logger.error(f"Token transfer analysis error: {e}")
            return {'token_address': token_address, 'error': str(e)}
    
    async def get_gas_price_trend(self, hours: int = 24) -> Dict:
        """가스 가격 추세 분석"""
        try:
            self.logger.info(f"가스 가격 트렌드 분석 시작 (기간: {hours}시간)")
            
            # 가스 가격 이력 가져오기
            gas_prices = [entry for entry in self.gas_price_history]
            
            if not gas_prices:
                self.logger.warning("가스 가격 이력이 없습니다. 현재 가스 가격 조회 시도")
                
                # 현재 가스 가격 조회
                try:
                    current_gas_price = await self.collector.async_web3.eth.gas_price
                    current_gwei = float(Web3.from_wei(current_gas_price, 'gwei'))
                    self.logger.info(f"현재 가스 가격: {current_gwei} Gwei")
                    
                    # 현재 가격을 이력에 추가
                    self.gas_price_history.append({
                        'timestamp': datetime.now(),
                        'gas_price_gwei': current_gwei
                    })
                    
                    return {
                        'trend': 'stable',
                        'average_gwei': current_gwei,
                        'current_gwei': current_gwei,
                        'change_percent': 0.0
                    }
                except Exception as e:
                    self.logger.error(f"현재 가스 가격 조회 오류: {e}")
                    # 기본값 반환
                    return {
                        'trend': 'unknown',
                        'average_gwei': 20.0,  # 일반적인 기본값
                        'current_gwei': 20.0,
                        'change_percent': 0.0
                    }
            
            # 시간대별 평균 계산
            hourly_prices = {}
            for entry in gas_prices:
                hour_key = entry['timestamp'].replace(minute=0, second=0, microsecond=0)
                if hour_key not in hourly_prices:
                    hourly_prices[hour_key] = []
                hourly_prices[hour_key].append(entry['gas_price_gwei'])
            
            hourly_avg = {
                hour: sum(prices) / len(prices)
                for hour, prices in hourly_prices.items()
            }
            
            # 추세 계산
            sorted_hours = sorted(hourly_avg.keys())
            if len(sorted_hours) < 2:
                return {
                    'trend': 'stable',
                    'average_gwei': hourly_avg[sorted_hours[0]]
                }
            
            # 첫/마지막 시간의 가격 비교
            first_price = hourly_avg[sorted_hours[0]]
            last_price = hourly_avg[sorted_hours[-1]]
            
            trend = 'stable'
            if last_price > first_price * 1.2:
                trend = 'increasing'
            elif last_price < first_price * 0.8:
                trend = 'decreasing'
            
            return {
                'trend': trend,
                'average_gwei': sum(hourly_avg.values()) / len(hourly_avg),
                'current_gwei': last_price,
                'change_percent': ((last_price - first_price) / first_price) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Gas price trend analysis error: {e}")
            # 기본값 반환
            return {
                'trend': 'unknown',
                'average_gwei': 20.0,
                'current_gwei': 20.0,
                'change_percent': 0.0
            }
    
    async def detect_abnormal_transactions(self) -> List[Dict]:
        """비정상 트랜잭션 감지"""
        try:
            recent_txs = list(self.analyzed_txs)
            if not recent_txs:
                return []
            
            abnormal_txs = []
            
            # 비정상적인 가스 비용
            gas_costs = [tx['gas_cost_eth'] for tx in recent_txs if 'gas_cost_eth' in tx]
            if gas_costs:
                gas_threshold = np.percentile(gas_costs, 95)
                
                for tx in recent_txs:
                    if tx.get('gas_cost_eth', 0) > gas_threshold:
                        abnormal_txs.append({
                            'tx_hash': tx['tx_hash'],
                            'abnormality': 'high_gas_cost',
                            'gas_cost_eth': tx['gas_cost_eth'],
                            'threshold': gas_threshold
                        })
            
            # 비정상적인 전송 금액
            eth_values = [tx['value_eth'] for tx in recent_txs if tx.get('value_eth', 0) > 0]
            if eth_values:
                value_threshold = np.percentile(eth_values, 95)
                
                for tx in recent_txs:
                    if tx.get('value_eth', 0) > value_threshold:
                        abnormal_txs.append({
                            'tx_hash': tx['tx_hash'],
                            'abnormality': 'high_value_transfer',
                            'value_eth': tx['value_eth'],
                            'threshold': value_threshold
                        })
            
            return abnormal_txs
            
        except Exception as e:
            self.logger.error(f"Abnormal transaction detection error: {e}")
            return []
    
    async def analyze_address_behavior(self, address: str) -> Dict:
        """주소 행동 패턴 분석"""
        try:
            if address not in self.address_activity:
                return {
                    'address': address,
                    'known': False,
                    'activity_level': 'unknown'
                }
            
            activity = self.address_activity[address]
            
            # 활동 수준 계산
            activity_level = 'low'
            if activity['tx_count'] > 100:
                activity_level = 'high'
            elif activity['tx_count'] > 20:
                activity_level = 'medium'
            
            # 컨트랙트 상호작용 비율
            contract_ratio = (
                activity['contract_interactions'] / activity['tx_count']
                if activity['tx_count'] > 0 else 0
            )
            
            # 활동 기간
            duration_days = (
                (activity['last_seen'] - activity['first_seen']).days
                if isinstance(activity['first_seen'], datetime) and 
                   isinstance(activity['last_seen'], datetime)
                else 0
            )
            
            return {
                'address': address,
                'known': True,
                'tx_count': activity['tx_count'],
                'total_eth_sent': activity['total_eth_sent'],
                'contract_interaction_ratio': contract_ratio,
                'activity_level': activity_level,
                'first_seen': activity['first_seen'],
                'last_seen': activity['last_seen'],
                'activity_duration_days': duration_days
            }
            
        except Exception as e:
            self.logger.error(f"Address behavior analysis error: {e}")
            return {'address': address, 'error': str(e)}