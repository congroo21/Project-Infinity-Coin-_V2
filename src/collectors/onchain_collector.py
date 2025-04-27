# src/collectors/onchain_collector.py

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import deque
from decimal import Decimal

from web3 import Web3, AsyncWeb3
from web3.exceptions import TransactionNotFound, BlockNotFound

from ..exceptions import ConnectionError, ValidationError, InsufficientDataError
from .base_collector import BaseCollector, BaseContent
from ..database import DatabaseManager

@dataclass
class BlockchainConfig:
    """블록체인 연결 설정"""
    rpc_url: str
    chain_id: int
    etherscan_api_key: Optional[str] = None
    max_blocks_per_request: int = 100
    update_interval: float = 10.0  # 10초마다 업데이트
    cache_size: int = 1000
    batch_size: int = 20
    explorer_api_url: Optional[str] = None
    subscription_topics: List[str] = None

@dataclass
class OnchainData(BaseContent):
    """온체인 데이터 클래스"""
    block_number: int = 0  # 기본값 추가
    block_hash: str = ""  # 기본값 추가
    transaction_count: int = 0  # 기본값 추가
    gas_used: int = 0  # 기본값 추가
    gas_limit: int = 0  # 기본값 추가
    base_fee_per_gas: Optional[float] = None
    difficulty: Optional[int] = None
    total_difficulty: Optional[int] = None
    extra_data: Optional[str] = None
    transactions: Optional[List[Dict]] = None
    
    def validate(self) -> bool:
        """데이터 유효성 검증"""
        if not super().validate():
            return False
        
        return all([
            isinstance(self.block_number, int) and self.block_number >= 0,
            isinstance(self.block_hash, str) and len(self.block_hash) > 0,
            isinstance(self.transaction_count, int) and self.transaction_count >= 0,
            isinstance(self.gas_used, int) and self.gas_used >= 0,
            isinstance(self.gas_limit, int) and self.gas_limit >= 0
        ])

class OnchainCollector(BaseCollector):
    """온체인 데이터 수집기"""
    def __init__(self, config: BlockchainConfig, db_manager: DatabaseManager):
        super().__init__(config.__dict__)
        self.blockchain_config = config
        self.db_manager = db_manager
        self.web3 = None
        self.async_web3 = None
        self.last_block_number = 0
        self.connected = False
        
        # 데이터 저장소
        self.block_cache = deque(maxlen=config.cache_size)
        self.transaction_cache = deque(maxlen=config.cache_size * 10)
        self.mempool_cache = deque(maxlen=config.cache_size)
        
        # 성능 모니터링
        self.request_count = 0
        self.error_count = 0
        self.retry_attempts = 3
        self.retry_delay = 0.5  # 초 단위
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Web3 연결 초기화"""
        try:
            # 동기 Web3 객체 초기화 (일부 호환성 문제를 위해 유지)
            self.web3 = Web3(Web3.HTTPProvider(self.blockchain_config.rpc_url))
            
            # PoA 미들웨어 사용 제거 - Web3.py 버전 호환성 문제 해결
            # 대신 특정 체인 ID에 따른 특별 처리가 필요하면 여기에 추가
            if self.web3.eth.chain_id in [56, 97, 137, 80001]:  # BSC, Polygon 등의 체인 ID
                self.logger.info(f"PoA 체인 감지됨 (Chain ID: {self.web3.eth.chain_id})")
            
            # 비동기 Web3 객체 초기화
            self.async_web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(self.blockchain_config.rpc_url))
            
            # 연결 테스트
            if not await self._test_connection():
                raise ConnectionError(f"Failed to connect to blockchain at {self.blockchain_config.rpc_url}")
                
            # 최신 블록 번호 가져오기
            self.last_block_number = await self.async_web3.eth.block_number
            self.connected = True
            
            self.logger.info(f"Successfully connected to blockchain. Current block: {self.last_block_number}")
            return True
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Failed to initialize blockchain connection: {e}")
            self.connected = False
            return False
    
    async def _test_connection(self) -> bool:
        """Web3 연결 테스트"""
        try:
            # 비동기 API로 체인 ID 확인
            chain_id = await self.async_web3.eth.chain_id
            
            # 설정된 체인 ID와 일치하는지 확인
            if chain_id != self.blockchain_config.chain_id:
                self.logger.warning(
                    f"Chain ID mismatch. Expected: {self.blockchain_config.chain_id}, Got: {chain_id}"
                )
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    async def collect(self) -> List[OnchainData]:
        """온체인 데이터 수집"""
        if not self.connected:
            if not await self.initialize():
                return []
        
        collected_data = []
        
        try:
            # 최신 블록 번호 가져오기
            current_block = await self.async_web3.eth.block_number
            self.logger.info(f"현재 블록 번호: {current_block}, 마지막 처리 블록: {self.last_block_number}")
            
            # 테스트를 위해 마지막 블록 번호 조정 (데이터 수집이 안되는 경우 무시)
            if self.last_block_number >= current_block:
                self.last_block_number = max(0, current_block - 5)
                self.logger.info(f"마지막 블록 번호를 {self.last_block_number}로 조정하여 데이터 수집 시도")
                
            # 일괄 처리를 위한 블록 범위 결정
            start_block = self.last_block_number + 1
            end_block = min(
                current_block, 
                start_block + min(5, self.blockchain_config.max_blocks_per_request - 1)  # 최대 5개로 제한
            )
            
            self.logger.info(f"Collecting blocks from {start_block} to {end_block}")
            
            # 블록 데이터 수집
            blocks_to_process = min(end_block - start_block + 1, 5)  # 최대 5개만 처리
            
            # 블록이 없으면 이전 블록부터 수집
            if blocks_to_process <= 0:
                self.logger.warning("수집할 새 블록이 없습니다. 최근 블록을 다시 수집합니다.")
                start_block = max(1, current_block - 3)
                end_block = current_block
                blocks_to_process = end_block - start_block + 1
                
            # 병렬로 블록 데이터 수집
            tasks = []
            for block_number in range(start_block, start_block + blocks_to_process):
                tasks.append(self._collect_block_data(block_number))
            
            # 병렬 실행 결과 수집
            block_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            successful_blocks = 0
            for result in block_results:
                if isinstance(result, Exception):
                    self.error_count += 1
                    self.logger.error(f"Block collection error: {result}")
                    continue
                    
                if result:
                    collected_data.append(result)
                    self.block_cache.append(result)
                    successful_blocks += 1
            
            # 마지막 처리 블록 업데이트
            if collected_data:
                new_last_block = max([data.block_number for data in collected_data])
                self.logger.info(f"{successful_blocks}개 블록 수집 완료, 마지막 블록 번호 {self.last_block_number} -> {new_last_block} 업데이트")
                self.last_block_number = new_last_block
            else:
                self.logger.warning("수집된 블록이 없습니다.")
            
            # 트랜잭션 풀(멤풀) 모니터링
            mempool_data = await self._monitor_mempool()
            
            return collected_data
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Block collection error: {e}")
            return []
    
    async def _collect_block_data(self, block_number: int) -> Optional[OnchainData]:
        """특정 블록의 데이터 수집"""
        for attempt in range(self.retry_attempts):
            try:
                # 블록 데이터 로깅 추가
                self.logger.info(f"블록 {block_number} 데이터 수집 시도 중 (시도: {attempt+1}/{self.retry_attempts})")
                
                # 블록 데이터 가져오기 (비동기)
                block = await self.async_web3.eth.get_block(block_number, full_transactions=False)
                
                if not block:
                    self.logger.warning(f"Block {block_number} not found")
                    return None
                
                # 트랜잭션 해시 목록
                tx_hashes = block.transactions
                self.logger.info(f"블록 {block_number} 수집 성공: {len(tx_hashes)}개 트랜잭션")
                
                # float 타입으로 변환하여 null 처리
                difficulty = getattr(block, 'difficulty', 0)
                if difficulty is None:
                    difficulty = 0
                    
                total_difficulty = getattr(block, 'totalDifficulty', 0)
                if total_difficulty is None:
                    total_difficulty = 0
                    
                base_fee = getattr(block, 'baseFeePerGas', None)
                if base_fee is not None:
                    base_fee = float(Web3.from_wei(base_fee, 'gwei'))
                
                # 기본 블록 정보 구성
                return OnchainData(
                    id=f"block_{block_number}",
                    timestamp=datetime.fromtimestamp(block.timestamp),
                    title=f"Block #{block_number}",
                    content=f"Block with {len(tx_hashes)} transactions",
                    source="blockchain",
                    block_number=block_number,
                    block_hash=block.hash.hex(),
                    transaction_count=len(tx_hashes),
                    gas_used=block.gasUsed,
                    gas_limit=block.gasLimit,
                    base_fee_per_gas=base_fee,
                    difficulty=difficulty,
                    total_difficulty=total_difficulty,
                    extra_data=block.extraData.hex(),
                    # 트랜잭션 해시 목록만 저장 (전체 트랜잭션은 별도 수집)
                    transactions=[tx.hex() for tx in tx_hashes]
                )
                    
            except BlockNotFound:
                self.logger.warning(f"Block {block_number} not found")
                return None
            except Exception as e:
                self.logger.warning(f"Attempt {attempt+1}: Error collecting block {block_number}: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        self.logger.error(f"Failed to collect block {block_number} after {self.retry_attempts} attempts")
        return None
    
    async def _monitor_mempool(self) -> Dict:
        """트랜잭션 풀(멤풀) 모니터링 - 개선된 버전"""
        try:
            # 'latest' 사용 대신 현재 블록 번호 직접 사용
            latest_block = await self.async_web3.eth.block_number
            
            # 수정된 코드: 
            # 특정 계정의 pending 트랜잭션 대신 전체 mempool 상태를 추정하는 방식 사용
            try:
                # 첫 번째 방법: JSON-RPC 직접 호출 시도
                try:
                    pending_count_result = await self.async_web3.provider.make_request("txpool_status", [])
                    if 'result' in pending_count_result and 'pending' in pending_count_result['result']:
                        pending_count = int(pending_count_result['result']['pending'], 16)
                        self.logger.info(f"txpool_status API로부터 보류 중인 트랜잭션 수 조회: {pending_count}")
                    else:
                        # 첫 번째 방법 실패 시 두 번째 방법 사용
                        raise Exception("txpool_status 결과에 'pending' 정보가 없습니다.")
                except Exception as e:
                    self.logger.info(f"txpool_status API 호출 실패: {e}, 대체 방법으로 전환합니다.")
                    # 두 번째 방법: 최근 블록 평균 트랜잭션 수 기반 추정
                    recent_blocks = list(self.block_cache)[-5:]  # 최근 5개 블록
                    if recent_blocks:
                        avg_tx_count = sum(block.transaction_count for block in recent_blocks) / len(recent_blocks)
                        pending_count = int(avg_tx_count * 1.5)  # 예상치로 평균 트랜잭션의 1.5배 사용
                        self.logger.info(f"최근 블록 평균 트랜잭션 기반 보류 중인 트랜잭션 수 추정: {pending_count} (원본: 평균 {avg_tx_count:.2f})")
                    else:
                        pending_count = 0
                        self.logger.warning("최근 블록 데이터가 없어 보류 중인 트랜잭션 수를 0으로 설정합니다.")
            except Exception as e:
                self.logger.warning(f"모든 방법으로 Pending 트랜잭션 수 조회 실패: {e}, 기본값 0을 사용합니다.")
                pending_count = 0
            
            # 가스 가격 조회
            gas_price = await self.async_web3.eth.gas_price
            
            mempool_info = {
                'timestamp': datetime.now(),
                'pending_transactions': pending_count,
                'gas_price_gwei': float(Web3.from_wei(gas_price, 'gwei')),
                'estimation_method': 'direct_api' if 'txpool_status' in locals() else 'block_average'
            }
            
            self.mempool_cache.append(mempool_info)
            return mempool_info
            
        except Exception as e:
            self.logger.error(f"Mempool monitoring error: {e}")
            # 오류 발생 시 기본값 반환
            return {
                'timestamp': datetime.now(),
                'pending_transactions': 0,
                'gas_price_gwei': 20.0,  # 기본 가스 가격 (gwei)
                'estimation_method': 'fallback_default'
            }
    
    async def get_transaction_details(self, tx_hash: str) -> Dict:
        """트랜잭션 상세 정보 조회"""
        try:
            # 트랜잭션 조회
            tx = await self.async_web3.eth.get_transaction(tx_hash)
            if not tx:
                return {}
                
            # 트랜잭션 영수증 조회 (성공 여부 확인)
            try:
                receipt = await self.async_web3.eth.get_transaction_receipt(tx_hash)
            except Exception as e:
                self.logger.warning(f"Failed to get transaction receipt: {e}")
                receipt = None
            
            # 결과 가공
            result = {
                'hash': tx_hash,
                'from': tx['from'],
                'to': tx.get('to', ''),  # 컨트랙트 생성은 to가 없을 수 있음
                'value': float(Web3.from_wei(tx['value'], 'ether')),
                'gas_price': float(Web3.from_wei(tx['gasPrice'], 'gwei')),
                'gas_used': receipt.get('gasUsed', 0) if receipt else 0,
                'status': receipt.get('status', None) if receipt else None,
                'block_number': tx['blockNumber'],
                'nonce': tx['nonce'],
                'timestamp': datetime.now()  # 실제로는 블록 타임스탬프로 대체 필요
            }
            
            return result
            
        except TransactionNotFound:
            self.logger.warning(f"Transaction {tx_hash} not found")
            return {}
        except Exception as e:
            self.logger.error(f"Error getting transaction details: {e}")
            return {}
    
    async def get_token_transfers(self, token_address: str, from_block: int, to_block: int) -> List[Dict]:
        """토큰 전송 이벤트 조회"""
        if not self.blockchain_config.etherscan_api_key:
            self.logger.warning("Etherscan API key not configured")
            return []
            
        try:
            # 토큰 전송 이벤트 필터 (ERC-20 Transfer 이벤트)
            transfer_event_signature = self.web3.keccak(
                text="Transfer(address,address,uint256)"
            ).hex()
            
            # 필터 생성
            filter_params = {
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': token_address,
                'topics': [transfer_event_signature]
            }
            
            # 이벤트 로그 조회
            logs = await self.async_web3.eth.get_logs(filter_params)
            
            transfers = []
            for log in logs:
                try:
                    # ERC-20 Transfer 이벤트 파싱
                    # Transfer(address indexed from, address indexed to, uint256 value)
                    decoded_log = {
                        'from': '0x' + log['topics'][1].hex()[-40:],
                        'to': '0x' + log['topics'][2].hex()[-40:],
                        'value': int(log['data'], 16),
                        'transaction_hash': log['transactionHash'].hex(),
                        'block_number': log['blockNumber'],
                        'log_index': log['logIndex']
                    }
                    transfers.append(decoded_log)
                except Exception as e:
                    self.logger.error(f"Error decoding transfer log: {e}")
            
            return transfers
            
        except Exception as e:
            self.logger.error(f"Error getting token transfers: {e}")
            return []
    
    async def get_whale_movements(self, min_value_eth: float = 100.0) -> List[Dict]:
        """대규모 거래(웨일 움직임) 모니터링"""
        try:
            # 최근 블록에서 대규모 ETH 전송 찾기
            recent_blocks = list(self.block_cache)
            if not recent_blocks:
                return []
                
            # 최근 100개 블록만 분석
            blocks_to_analyze = sorted(
                recent_blocks, 
                key=lambda x: x.block_number, 
                reverse=True
            )[:100]
            
            whale_movements = []
            min_value_wei = Web3.to_wei(min_value_eth, 'ether')
            
            for block_data in blocks_to_analyze:
                if not block_data.transactions:
                    continue
                    
                # 블록의 트랜잭션 중 일부만 샘플링하여 분석 (전체 분석은 리소스 부담)
                sample_size = min(20, len(block_data.transactions))
                sampled_txs = block_data.transactions[:sample_size]
                
                for tx_hash in sampled_txs:
                    tx_details = await self.get_transaction_details(tx_hash)
                    if not tx_details:
                        continue
                        
                    # 대규모 거래 식별
                    value_wei = Web3.to_wei(tx_details['value'], 'ether')
                    if value_wei >= min_value_wei:
                        whale_movements.append(tx_details)
            
            return whale_movements
            
        except Exception as e:
            self.logger.error(f"Error monitoring whale movements: {e}")
            return []
    
    async def get_contract_events(self, contract_address: str, event_signature: str, from_block: int, to_block: int) -> List[Dict]:
        """스마트 컨트랙트 이벤트 조회"""
        try:
            # 이벤트 시그니처 해시
            event_hash = self.web3.keccak(text=event_signature).hex()
            
            # 이벤트 필터
            filter_params = {
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': contract_address,
                'topics': [event_hash]
            }
            
            # 이벤트 로그 조회
            logs = await self.async_web3.eth.get_logs(filter_params)
            
            # 로그 정보 가공
            events = []
            for log in logs:
                event_data = {
                    'transaction_hash': log['transactionHash'].hex(),
                    'block_number': log['blockNumber'],
                    'log_index': log['logIndex'],
                    'data': log['data'],
                    'topics': [t.hex() for t in log['topics']]
                }
                events.append(event_data)
                
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting contract events: {e}")
            return []
    
    async def get_onchain_metrics(self) -> Dict:
        """온체인 메트릭스 수집"""
        try:
            # 가스 가격
            gas_price = await self.async_web3.eth.gas_price
            gas_price_gwei = float(Web3.from_wei(gas_price, 'gwei'))
            
            # 블록 생성 속도 (최근 100개 블록 기준)
            recent_blocks = sorted(
                list(self.block_cache),
                key=lambda x: x.block_number
            )[-100:]
            
            if len(recent_blocks) >= 2:
                first_block = recent_blocks[0]
                last_block = recent_blocks[-1]
                
                # 블록 넘버 차이
                blocks_diff = last_block.block_number - first_block.block_number
                
                # 시간 차이 (초 단위)
                time_diff = (last_block.timestamp - first_block.timestamp).total_seconds()
                
                # 블록 생성 시간 (초/블록)
                block_time = time_diff / max(1, blocks_diff) if blocks_diff > 0 else 0
            else:
                block_time = 0
            
            # 트랜잭션 수 집계
            tx_count = sum(b.transaction_count for b in recent_blocks) if recent_blocks else 0
            avg_tx_per_block = tx_count / len(recent_blocks) if recent_blocks else 0
            
            # 네트워크 활성도 지표
            network_activity = {
                'block_time_seconds': block_time,
                'avg_transactions_per_block': avg_tx_per_block,
                'gas_price_gwei': gas_price_gwei,
                'mempool_size': self.mempool_cache[-1].get('pending_transactions', 0) if self.mempool_cache else 0
            }
            
            return network_activity
            
        except Exception as e:
            self.logger.error(f"Error getting onchain metrics: {e}")
            return {}
    
    async def cleanup(self):
        """리소스 정리 (상속된 메서드 오버라이드)"""
        self.logger.info("Cleaning up onchain collector resources")
        
        # 캐시 정리
        self.block_cache.clear()
        self.transaction_cache.clear()
        self.mempool_cache.clear()