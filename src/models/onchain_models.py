# src/models/onchain_models.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

@dataclass
class BlockData:
    """블록 데이터 모델"""
    block_number: int
    timestamp: datetime
    transactions_count: int
    gas_used: int
    gas_limit: int
    base_fee_per_gas: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'block_number': self.block_number,
            'timestamp': self.timestamp.isoformat(),
            'transactions_count': self.transactions_count,
            'gas_used': self.gas_used,
            'gas_limit': self.gas_limit,
            'base_fee_per_gas': self.base_fee_per_gas
        }

@dataclass
class TransactionData:
    """트랜잭션 데이터 모델"""
    tx_hash: str
    block_number: int
    timestamp: datetime
    from_address: str
    to_address: Optional[str]
    value: float  # ETH 단위
    gas_price: float  # Gwei 단위
    gas_used: int
    status: bool  # 성공 여부
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'tx_hash': self.tx_hash,
            'block_number': self.block_number,
            'timestamp': self.timestamp.isoformat(),
            'from_address': self.from_address,
            'to_address': self.to_address,
            'value': self.value,
            'gas_price': self.gas_price,
            'gas_used': self.gas_used,
            'status': self.status
        }

@dataclass
class TokenTransferData:
    """토큰 전송 데이터 모델"""
    tx_hash: str
    block_number: int
    timestamp: datetime
    token_address: str
    from_address: str
    to_address: str
    value: float
    token_symbol: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'tx_hash': self.tx_hash,
            'block_number': self.block_number,
            'timestamp': self.timestamp.isoformat(),
            'token_address': self.token_address,
            'from_address': self.from_address,
            'to_address': self.to_address,
            'value': self.value,
            'token_symbol': self.token_symbol
        }

@dataclass
class BlockchainMetrics:
    """블록체인 메트릭스 모델"""
    timestamp: datetime
    block_time: float  # 초 단위
    gas_price: float  # Gwei 단위
    tx_per_block: float
    mempool_size: int
    network_congestion: float  # 0-1 사이 값 (네트워크 혼잡도)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'block_time': self.block_time,
            'gas_price': self.gas_price,
            'tx_per_block': self.tx_per_block,
            'mempool_size': self.mempool_size,
            'network_congestion': self.network_congestion
        }

@dataclass
class WhaleMovement:
    """대규모 거래(웨일 움직임) 모델"""
    tx_hash: str
    block_number: int
    timestamp: datetime
    from_address: str
    to_address: str
    value: float  # ETH 단위
    category: str  # 'exchange_deposit', 'exchange_withdrawal', 'wallet_transfer' 등
    impact_score: float  # 0-1 사이 값 (예상 시장 영향도)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'tx_hash': self.tx_hash,
            'block_number': self.block_number,
            'timestamp': self.timestamp.isoformat(),
            'from_address': self.from_address,
            'to_address': self.to_address,
            'value': self.value,
            'category': self.category,
            'impact_score': self.impact_score
        }

@dataclass
class MarketSignal:
    """온체인 데이터 기반 시장 신호 모델"""
    timestamp: datetime
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-1 사이 값
    time_horizon: str  # 'short', 'medium', 'long'
    confidence: float  # 0-1 사이 값
    source: str  # 'whale_movement', 'exchange_flow', 'network_activity' 등
    related_assets: List[str]  # 관련 자산 목록
    supporting_data: Dict  # 근거 데이터
    predicted_impact: Dict[str, float]  # 자산별 예상 가격 영향 (%)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'signal_type': self.signal_type,
            'strength': self.strength,
            'time_horizon': self.time_horizon,
            'confidence': self.confidence,
            'source': self.source,
            'related_assets': self.related_assets,
            'supporting_data': self.supporting_data,
            'predicted_impact': self.predicted_impact
        }

@dataclass
class NetworkState:
    """블록체인 네트워크 상태 모델"""
    timestamp: datetime
    chain_id: int
    block_height: int
    is_congested: bool
    average_block_time: float
    average_gas_price: float
    active_addresses_24h: int
    transaction_count_24h: int
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'chain_id': self.chain_id,
            'block_height': self.block_height,
            'is_congested': self.is_congested,
            'average_block_time': self.average_block_time,
            'average_gas_price': self.average_gas_price,
            'active_addresses_24h': self.active_addresses_24h,
            'transaction_count_24h': self.transaction_count_24h
        }