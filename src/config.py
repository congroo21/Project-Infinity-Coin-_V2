# src/config.py 상단에 import 추가

from pathlib import Path
import os
from dotenv import load_dotenv
from typing import Optional  # Optional import 추가
from .exceptions import TradingSystemError  # TradingSystemError import 추가

# .env 파일 로드
load_dotenv()

# 기본 설정
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

class Config:
    """설정 클래스"""
    # API 설정
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4o"
    
    # 데이터베이스 설정
    DB_PATH = str(BASE_DIR / "data" / "trading.db")
    db_path = DB_PATH  # db_path 속성 추가 (소문자로)
    
    # 트레이딩 설정
    TRADE_INTERVAL = 0.01
    MAX_POSITION_SIZE = 0.1  # 10%
    EMERGENCY_STOP_LOSS = 0.02  # 2%
    MIN_TRADE_AMOUNT = 0.001  # 최소 거래량
    TRADE_FEE = 0.0005  # 거래 수수료 (0.05%)
    
    # 시장 분석 설정
    TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
    ANALYSIS_INTERVAL = 1  # 1초
    MIN_DATA_POINTS = {
        '1m': 100,
        '5m': 50,
        '15m': 30,
        '1h': 24,
        '4h': 10,
        '1d': 7
    }
    
    # 리스크 관리 설정
    RISK_THRESHOLDS = {
        'volatility_normal': 0.02,    # 정상 상태 (2% 이하)
        'volatility_warning': 0.05,   # 경고 상태 (5% 이하) - 이전 'volatility_high'(0.03)에서 상향 조정
        'volatility_restricted': 0.08, # 제한 상태 (8% 이하)
        'volatility_emergency': 0.12, # 비상 상태 (12% 이상)
        'trend_strong': 0.02,
        'liquidity_high': 1000000,
        'max_drawdown': 0.02,
        'position_limit': 0.1,
        'concentration_limit': 0.25
    }
    
    # 시스템 모니터링
    MONITORING_INTERVAL = 1  # 1초
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY = 1  # 1초
    SYSTEM_METRICS = {
        'cpu_threshold': 80,  # CPU 사용률 임계값 (%)
        'memory_threshold': 85,  # 메모리 사용률 임계값 (%)
        'disk_threshold': 90  # 디스크 사용률 임계값 (%)
    }
    
    # 캐시 설정
    CACHE_CONFIG = {
        'price_cache_size': 1000,
        'orderbook_cache_size': 100,
        'analysis_cache_size': 500,
        'cache_ttl': 60  # 캐시 유효 시간 (초)
    }

    @classmethod
    def validate(cls) -> bool:
        """설정 유효성 검증"""
        required_env_vars = ['OPENAI_API_KEY']
        return all(os.getenv(var) for var in required_env_vars)

    @classmethod
    def get_db_url(cls) -> str:
        """데이터베이스 URL 생성"""
        return f"sqlite:///{cls.DB_PATH}"
    

class InsufficientDataError(TradingSystemError):
    def __init__(self, message: str = "데이터 부족", min_required: Optional[int] = None):
        if min_required:
            message = f"{message} (최소 필요 데이터 포인트: {min_required})"
        super().__init__(message)
        self.min_required = min_required

# 설정 유효성 검증
if not Config.validate():
    raise ValueError("필수 환경 변수가 설정되지 않았습니다.")

ONCHAIN_CONFIG = {
    'enable_onchain_analysis': True,
    'update_interval': 60,  # 60초마다 업데이트
    'whale_threshold_eth': 100.0  # 100 ETH 이상을 웨일로 간주
}