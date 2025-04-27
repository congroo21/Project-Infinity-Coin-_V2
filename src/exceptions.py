# src/xceptions.py

class TradingSystemError(Exception):
    """트레이딩 시스템의 기본 예외 클래스
    
    모든 트레이딩 시스템 관련 예외의 기본이 되는 클래스입니다.
    """
    def __init__(self, message: str = "트레이딩 시스템 오류"):
        self.message = message
        super().__init__(self.message)

class MarketDataError(TradingSystemError):
    """시장 데이터 관련 예외
    
    시장 데이터 수집, 처리, 분석 과정에서 발생하는 오류를 처리합니다.
    """
    def __init__(self, message: str = "시장 데이터 오류"):
        super().__init__(message)

class RiskLimitExceededError(TradingSystemError):
    """리스크 한도 초과 예외
    
    설정된 리스크 한도를 초과했을 때 발생하는 오류를 처리합니다.
    """
    def __init__(self, message: str = "리스크 한도 초과"):
        super().__init__(message)

class DatabaseError(TradingSystemError):
    """데이터베이스 관련 예외
    
    데이터베이스 연결, 쿼리, 저장 등의 작업에서 발생하는 오류를 처리합니다.
    """
    def __init__(self, message: str = "데이터베이스 오류"):
        super().__init__(message)

class APIError(TradingSystemError):
    """API 관련 예외
    
    외부 API 호출 및 응답 처리 과정에서 발생하는 오류를 처리합니다.
    """
    def __init__(self, message: str = "API 오류"):
        super().__init__(message)

class ValidationError(TradingSystemError):
    """데이터 검증 관련 예외
    
    입력 데이터나 시스템 상태의 유효성 검증 실패 시 발생하는 오류를 처리합니다.
    """
    def __init__(self, message: str = "데이터 검증 오류"):
        super().__init__(message)

class ConfigurationError(TradingSystemError):
    """설정 관련 예외
    
    시스템 설정이나 환경 설정과 관련된 오류를 처리합니다.
    """
    def __init__(self, message: str = "설정 오류"):
        super().__init__(message)

class SystemResourceError(TradingSystemError):
    """시스템 리소스 관련 예외
    
    메모리, CPU 등 시스템 리소스 부족 관련 오류를 처리합니다.
    """
    def __init__(self, message: str = "시스템 리소스 부족"):
        super().__init__(message)

class TradeExecutionError(TradingSystemError):
    """거래 실행 관련 예외
    
    주문 실행, 취소 등 거래 관련 작업에서 발생하는 오류를 처리합니다.
    """
    def __init__(self, message: str = "거래 실행 오류"):
        super().__init__(message)

class PositionLimitError(TradingSystemError):
    """포지션 한도 관련 예외
    
    설정된 포지션 한도를 초과했을 때 발생하는 오류를 처리합니다.
    """
    def __init__(self, message: str = "포지션 한도 초과"):
        super().__init__(message)

class InsufficientDataError(TradingSystemError):
    """데이터 부족 관련 예외
    
    분석이나 거래에 필요한 데이터가 부족할 때 발생하는 오류를 처리합니다.
    """
    def __init__(self, message: str = "데이터 부족", min_required: int = None):
        if min_required is not None:
            message = f"{message} (최소 필요 데이터 포인트: {min_required})"
        super().__init__(message)
        self.min_required = min_required

class EmergencyShutdownError(TradingSystemError):
    """긴급 종료 관련 예외
    
    시스템 긴급 종료가 필요한 상황에서 발생하는 오류를 처리합니다.
    """
    def __init__(self, message: str = "긴급 시스템 종료"):
        super().__init__(message)

class SystemEmergencyError(TradingSystemError):
    """시스템 긴급 상황 예외
    
    시스템의 긴급 상황이나 비정상적인 상태를 처리합니다.
    """
    def __init__(self, message: str = "시스템 긴급 상황 발생"):
        super().__init__(message)

class TimeoutError(TradingSystemError):
    """시간 초과 예외
    
    특정 작업이 제한 시간 내에 완료되지 않았을 때 발생하는 오류를 처리합니다.
    """
    def __init__(self, message: str = "작업 시간 초과"):
        super().__init__(message)

class ConnectionError(TradingSystemError):
    """연결 관련 예외
    
    네트워크 연결이나 외부 시스템과의 통신 오류를 처리합니다.
    """
    def __init__(self, message: str = "연결 오류 발생"):
        super().__init__(message)

class OrderbookError(TradingSystemError):
    """호가창 관련 예외
    
    호가창 데이터 수집 및 처리 과정에서 발생하는 오류를 처리합니다.
    """
    def __init__(self, message: str = "호가창 데이터 오류"):
        super().__init__(message)

    class InsufficientDataError(TradingSystemError):
        """데이터 부족 예외 클래스 수정"""
        def __init__(self, message: str = "데이터 부족", min_required: int = None):
            if min_required:
                message = f"{message} (최소 필요 데이터 포인트: {min_required})"
            super().__init__(message)
            self.min_required = min_required

# src/exceptions.py에 새 예외 클래스 추가

class RiskWarningError(TradingSystemError):
    """리스크 경고 상태 예외"""
    def __init__(self, message: str = "리스크 경고 상태"):
        super().__init__(message)

class RiskRestrictedError(TradingSystemError):
    """리스크 제한 상태 예외"""
    def __init__(self, message: str = "리스크 제한 상태"):
        super().__init__(message)

# 하위 호환성을 위한 별칭들
RiskLimitExceeded = RiskLimitExceededError
SystemEmergency = SystemEmergencyError