# File: src/scenarios/risk/risk_manager.py

# 기본 Python 라이브러리
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# 수치 계산 및 통계
import numpy as np
from scipy import stats

# 커스텀 모듈
from src.database import ImprovedDatabaseManager
from src.exceptions import RiskLimitExceededError, SystemEmergencyError
# 변동성 유틸리티 가져오기
from src.utils.volatility_utils import calculate_volatility

@dataclass
class RiskThresholds:
    """리스크 임계값 설정"""
    max_position_size: float = 0.1          # 최대 포지션 크기 (10%)
    max_drawdown: float = 0.02              # 최대 허용 손실폭 (2%)
    # 변동성 임계값 현실화 - 단계별 임계값 추가
    volatility_normal: float = 0.02         # 정상 상태 (2% 이하)
    volatility_warning: float = 0.05        # 경고 상태 (5% 이하) - 이전 'volatility_high'(0.03)에서 상향 조정
    volatility_restricted: float = 0.08     # 제한 상태 (8% 이하)
    volatility_emergency: float = 0.12      # 비상 상태 (12% 이상)
    volatility_threshold: float = 0.05      # 하위 호환성 유지 (이전 코드와의 호환성)
    liquidity_threshold: float = 1000000    # 최소 유동성 (100만원)
    max_trade_count: int = 1000             # 단위 시간당 최대 거래 횟수
    var_limit: float = 0.02                 # VaR 한도 (2%)
    correlation_threshold: float = 0.7      # 상관관계 임계값
    concentration_limit: float = 0.25       # 집중도 한도 (25%)

@dataclass
class RiskState:
    """리스크 상태 데이터"""
    current_positions: Dict = field(default_factory=dict)
    current_exposure: float = 0.0
    risk_metrics: Dict = field(default_factory=dict)
    var_levels: Dict = field(default_factory=dict)
    last_update: Optional[datetime] = None
    emergency_mode: bool = False
    risk_level: str = 'normal'  # 'normal', 'warning', 'restricted', 'emergency'
    warning_level: str = 'normal'  # 하위 호환성 유지
    active_alerts: List = field(default_factory=list)
    timeframe_metrics: Dict = field(default_factory=dict)  # 시간프레임별 메트릭스 추가

class ImprovedRiskManager:
    """개선된 리스크 관리 시스템
       - 포트폴리오 비중 자동 조정 기능
       - 시장 변동성 급증 시 포지션 크기 자동 조정
       - 손실 제한 기능(특정 기준 초과 시 자동 청산) 추가
       - **동적 리스크 조정**: 기존 정적 방식 대신 RL 기반 동적 위험 조정 계수를 산출하여 allowed_size에 반영함.
    """
    def __init__(self, config: Dict = None):
        # 기본 설정
        self.thresholds = RiskThresholds()
        self.state = RiskState()
        
        # 데이터베이스 연결
        self.db = ImprovedDatabaseManager()
        
        # 모니터링 설정
        self.monitoring_interval = 1  # 1초
        self.cleanup_interval = 3600  # 1시간
        
        # 성능 추적
        self.performance_history = []
        self.risk_event_history = []
        
        # 실행 상태
        self.running = False
        self.last_check = datetime.now()
        
        # 설정 업데이트
        if config:
            self._update_config(config)

    def _update_config(self, config: Dict):
        """설정 업데이트"""
        try:
            if 'max_position_size' in config:
                self.thresholds.max_position_size = config['max_position_size']
            if 'max_drawdown' in config:
                self.thresholds.max_drawdown = config['max_drawdown']
            # 변동성 임계값 설정 업데이트
            if 'volatility_normal' in config:
                self.thresholds.volatility_normal = config['volatility_normal']
            if 'volatility_warning' in config:
                self.thresholds.volatility_warning = config['volatility_warning']
                self.thresholds.volatility_threshold = config['volatility_warning']  # 호환성 유지
            if 'volatility_restricted' in config:
                self.thresholds.volatility_restricted = config['volatility_restricted']
            if 'volatility_emergency' in config:
                self.thresholds.volatility_emergency = config['volatility_emergency']
            if 'var_limit' in config:
                self.thresholds.var_limit = config['var_limit']
            
            logging.info("Risk manager configuration updated successfully")
            
        except Exception as e:
            logging.error(f"Error updating risk manager config: {e}")
            raise ValueError(f"Invalid configuration: {str(e)}")

    async def start_monitoring(self):
        """리스크 모니터링 시작"""
        self.running = True
        logging.info("Starting risk monitoring system...")
        
        try:
            tasks = [
                asyncio.create_task(self._risk_monitoring_loop()),
                asyncio.create_task(self._cleanup_loop())
            ]
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logging.error(f"Error in risk monitoring: {e}")
            self.running = False
            raise SystemEmergencyError("Risk monitoring system failed")

    async def _risk_monitoring_loop(self):
        """주요 리스크 모니터링 루프"""
        while self.running:
            try:
                # 시장 데이터 수집
                market_data = await self._get_market_data()
                if not market_data:
                    await asyncio.sleep(self.monitoring_interval)
                    continue

                # 리스크 메트릭스 계산
                risk_metrics = await self._calculate_risk_metrics(market_data)
                
                # VaR 계산
                var_levels = self._calculate_var_levels(market_data)
                
                # 리스크 상태 업데이트
                self.state.risk_metrics = risk_metrics
                self.state.var_levels = var_levels
                self.state.last_update = datetime.now()

                # 리스크 한도 체크
                await self._check_risk_limits(risk_metrics, var_levels, market_data)
                
                # 긴급 상황 체크
                if self._check_emergency_conditions(risk_metrics):
                    await self._handle_emergency_situation()

                # ───────────────────────────────
                # [추가] 포트폴리오 자동 조정 기능 (동적 RL 기반 조정 포함):
                await self._auto_adjust_portfolio(risk_metrics)
                
                # [추가] 손실 제한 기능 (자동 청산):
                await self._apply_stop_loss(market_data)
                # ───────────────────────────────

                # DB에 저장
                await self._save_risk_state()
                
                # 모니터링 간격 대기
                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logging.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(1)  # 에러 발생시 더 긴 대기

    async def _cleanup_loop(self):
        """정리 작업 루프"""
        while self.running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                logging.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # 에러 발생시 1분 대기

    async def check_trade(self, trade_data: Dict) -> Dict:
        """거래 실행 전 리스크 체크"""
        try:
            # 기본 리스크 체크
            position_check = self._check_position_limits(trade_data)
            var_check = self._check_var_limits(trade_data)
            liquidity_check = await self._check_liquidity(trade_data)

            # 모든 체크 통과 확인
            is_safe = all([
                position_check['is_safe'],
                var_check['is_safe'],
                liquidity_check['is_safe']
            ])

            # 리스크 레벨에 따른 추가 검사
            if self.state.risk_level != 'normal':
                # 'warning' 상태에서는 경고만 기록
                if self.state.risk_level == 'warning':
                    logging.warning(f"Trading under warning risk level: {trade_data['symbol']}")
                # 'restricted' 상태에서는 포지션 크기 제한
                elif self.state.risk_level == 'restricted':
                    position_check['allowed_size'] *= 0.5  # 50% 크기 제한
                    logging.warning(f"Position size restricted due to risk level: {trade_data['symbol']}")
                # 'emergency' 상태에서는 거래 금지
                elif self.state.risk_level == 'emergency':
                    is_safe = False
                    logging.error(f"Trade rejected due to emergency risk level: {trade_data['symbol']}")
                    return {'is_safe': False, 'reason': "Emergency risk level active"}

            if not is_safe:
                reasons = []
                if not position_check['is_safe']:
                    reasons.append(position_check['reason'])
                if not var_check['is_safe']:
                    reasons.append(var_check['reason'])
                if not liquidity_check['is_safe']:
                    reasons.append(liquidity_check['reason'])

                raise RiskLimitExceededError(", ".join(reasons))

            # 정적 allowed_size 값 산출 후, RL 기반 동적 리스크 조정 계수를 반영하여 최종 allowed_size 결정
            static_allowed = min(
                position_check['allowed_size'],
                var_check['allowed_size'],
                liquidity_check['allowed_size']
            )
            dynamic_factor = self._rl_dynamic_adjustment(self.state.risk_metrics)
            return {
                'is_safe': True,
                'adjusted_size': static_allowed * dynamic_factor
            }

        except RiskLimitExceededError as e:
            logging.warning(f"Trade rejected due to risk limits: {e}")
            return {'is_safe': False, 'reason': str(e)}
        except Exception as e:
            logging.error(f"Error in trade risk check: {e}")
            return {'is_safe': False, 'reason': f"Risk check error: {str(e)}"}

    async def update_position(self, position_data: Dict):
        """포지션 업데이트"""
        try:
            symbol = position_data['symbol']
            
            if symbol in self.state.current_positions:
                # 기존 포지션 업데이트
                current = self.state.current_positions[symbol]
                current['size'] = position_data['size']
                current['last_update'] = datetime.now()
            else:
                # 새 포지션 추가
                self.state.current_positions[symbol] = {
                    'size': position_data['size'],
                    'entry_price': position_data['price'],
                    'last_update': datetime.now()
                }

            # 총 익스포저 업데이트
            self.state.current_exposure = sum(
                pos['size'] * pos.get('entry_price', 0)
                for pos in self.state.current_positions.values()
            )

            # 리스크 메트릭스 업데이트
            await self._update_risk_metrics()

        except Exception as e:
            logging.error(f"Error updating position: {e}")
            raise ValueError(f"Position update failed: {str(e)}")

# src/scenarios/risk/risk_manager.py의 관련 함수 수정

    async def _calculate_risk_metrics(self, market_data: Dict) -> Dict:
        """리스크 메트릭스 계산"""
        try:
            # 수익률 데이터 가져오기
            returns = market_data.get('returns', [])
            
            if len(returns) < 2:
                return {}
                
            # 중앙화된 리스크 계산기 사용
            from src.utils.risk_utils import risk_calculator
            
            # 종합 리스크 메트릭스 계산
            risk_metrics = risk_calculator.calculate_risk_metrics(returns)
            
            # 상관관계 계산
            correlations = self._calculate_correlations(market_data)
            
            # 집중도 계산
            concentration = self._calculate_concentration()
            
            # 결과 통합
            risk_metrics.update({
                'correlations': correlations,
                'concentration': concentration,
                'timestamp': datetime.now()
            })
            
            return risk_metrics
            
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {e}")
            return {}

    def _calculate_var_levels(self, market_data: Dict) -> Dict:
        """VaR (Value at Risk) 계산"""
        try:
            returns = market_data.get('returns', [])
            
            if len(returns) < 2:
                return {}
            
            # 중앙화된 리스크 계산기 사용
            from src.utils.risk_utils import risk_calculator
            
            # VaR 계산
            var_levels = risk_calculator.calculate_var_levels(returns)
            
            return var_levels
            
        except Exception as e:
            logging.error(f"Error calculating VaR: {e}")
            return {}

    def _calculate_correlations(self, market_data: Dict) -> Dict:
        """자산 간 상관관계 계산"""
        try:
            if 'asset_prices' not in market_data:
                return {}
                
            prices = market_data['asset_prices']
            if not prices or len(prices) < 2:
                return {}
                
            correlation_matrix = np.corrcoef(prices)
            assets = market_data.get('asset_names', [])
            correlations = {}
            
            for i, asset1 in enumerate(assets):
                correlations[asset1] = {}
                for j, asset2 in enumerate(assets):
                    correlations[asset1][asset2] = correlation_matrix[i, j]
                    
            return correlations
            
        except Exception as e:
            logging.error(f"Error calculating correlations: {e}")
            return {}

    def _calculate_concentration(self) -> float:
        """포트폴리오 집중도 계산"""
        try:
            total_value = sum(pos['size'] * pos.get('entry_price', 0)
                              for pos in self.state.current_positions.values())
            
            if total_value == 0:
                return 0.0
                
            position_weights = [
                (pos['size'] * pos.get('entry_price', 0)) / total_value
                for pos in self.state.current_positions.values()
            ]
            
            # Herfindahl-Hirschman Index (HHI)
            return sum(w * w for w in position_weights)
            
        except Exception as e:
            logging.error(f"Error calculating concentration: {e}")
            return 0.0

    async def _check_risk_limits(self, risk_metrics: Dict, var_levels: Dict, market_data: Dict):
        """리스크 한도 체크 - 단계적 대응 추가"""
        try:
            # 이전 경고 로직 유지 (호환성)
            warnings = []
            
            # 변동성 체크
            if risk_metrics.get('volatility', 0) > self.thresholds.volatility_warning:
                warnings.append(f"High volatility: {risk_metrics['volatility']:.4f}")
            
            # VaR 체크
            if var_levels.get('parametric_var_95', 0) > self.thresholds.var_limit:
                warnings.append(f"VaR exceeded limit: {var_levels['parametric_var_95']:.4f}")
            
            # 집중도 체크
            if risk_metrics.get('concentration', 0) > self.thresholds.concentration_limit:
                warnings.append(f"High concentration: {risk_metrics['concentration']:.4f}")
            
            if warnings:
                await self._handle_risk_warnings(warnings)
            
            # 시간프레임별 리스크 분석
            await self._analyze_risk_by_timeframes(market_data)
            
            # 리스크 레벨 결정
            risk_level = await self._determine_risk_level()
            
            # 리스크 레벨에 따른 대응
            await self._handle_risk_level(risk_level)
                
        except Exception as e:
            logging.error(f"리스크 한도 체크 오류: {e}")

    async def _analyze_risk_by_timeframes(self, market_data: Dict) -> Dict:
        """시간 프레임별 리스크 분석"""
        try:
            # 여러 시간프레임 데이터 가져오기
            timeframes = {
                'short': self._get_timeframe_data(market_data, '1m'),
                'medium': self._get_timeframe_data(market_data, '5m'),
                'long': self._get_timeframe_data(market_data, '15m')
            }
            
            # 시간프레임별 변동성 계산 - 중앙화된 유틸리티 함수 사용
            volatilities = {
                tf_name: calculate_volatility(tf_data) 
                for tf_name, tf_data in timeframes.items()
            }
            
            # 가중 평균 계산
            weighted_vol = (
                volatilities.get('short', 0) * 0.5 +
                volatilities.get('medium', 0) * 0.3 +
                volatilities.get('long', 0) * 0.2
            )
            
            # 결과 저장
            self.state.timeframe_metrics = {
                'volatilities': volatilities,
                'weighted_volatility': weighted_vol
            }
            
            return volatilities
            
        except Exception as e:
            logging.error(f"시간프레임별 리스크 분석 오류: {e}")
            return {}
    
    def _get_timeframe_data(self, market_data: Dict, timeframe: str) -> List[float]:
        """특정 시간프레임의 가격 데이터 추출"""
        # 실제 구현 시 market_data에서 해당 시간프레임 데이터 추출
        # 예시 데이터 반환
        return market_data.get(f'prices_{timeframe}', market_data.get('prices', [100.0, 101.0, 102.0, 101.5, 102.5]))

    async def _determine_risk_level(self) -> str:
        """리스크 수준 결정 - 단계적 대응을 위한 메서드"""
        try:
            # 가중 변동성 가져오기
            weighted_vol = self.state.timeframe_metrics.get('weighted_volatility', 0)
            
            # 리스크 레벨 결정 - 명확한 임계값 참조
            if weighted_vol >= self.thresholds.volatility_emergency:
                return 'emergency'
            elif weighted_vol >= self.thresholds.volatility_restricted:
                return 'restricted'
            elif weighted_vol >= self.thresholds.volatility_warning:
                return 'warning'
            else:
                return 'normal'
                
        except Exception as e:
            logging.error(f"리스크 레벨 결정 오류: {e}")
            return 'normal'  # 오류 발생 시 기본값
    
    async def _handle_risk_level(self, risk_level: str):
        """리스크 레벨에 따른 대응"""
        try:
            # 이전 상태와 비교해 변경되었을 때만 처리
            if risk_level == self.state.risk_level:
                return
                
            # 상태 업데이트
            previous_level = self.state.risk_level
            self.state.risk_level = risk_level
            # 하위 호환성 유지
            self.state.warning_level = 'high' if risk_level in ['restricted', 'emergency'] else ('medium' if risk_level == 'warning' else 'normal')
            
            logging.info(f"리스크 레벨 변경: {previous_level} -> {risk_level}")
            
            # 리스크 레벨별 대응
            if risk_level == 'warning':
                await self._handle_warning_state()
            elif risk_level == 'restricted':
                await self._handle_restricted_state()
            elif risk_level == 'emergency':
                await self._handle_emergency_state()
            else:  # normal
                await self._handle_normal_state()
                
        except Exception as e:
            logging.error(f"리스크 레벨 처리 오류: {e}")
    
    async def _handle_normal_state(self):
        """정상 상태 처리"""
        logging.info("정상 거래 상태로 복귀")
        # 정상 운영 상태로 복귀
        self.state.emergency_mode = False
        
    async def _handle_warning_state(self):
        """경고 상태 처리"""
        logging.warning("경고 상태 진입: 거래 계속되지만 모니터링 강화")
        # 로깅 강화, 모니터링 주기 단축 등
        # 모니터링 주기 단축 (기존의 절반)
        self.monitoring_interval = max(0.5, self.monitoring_interval / 2)  # 최소 0.5초
        
        # 리스크 이벤트 저장
        await self.db.save_risk_event({
            'event_type': 'warning',
            'severity': 'medium',
            'description': '시장 변동성 증가로 경고 상태 진입',
            'action_taken': '모니터링 강화'
        })
        
    async def _handle_restricted_state(self):
        """제한 상태 처리"""
        logging.warning("제한 상태 진입: 거래 규모 제한")
        # 포지션 크기 축소, 신규 거래 제한 등
        
        # 기존 포지션 축소 (예: 50%)
        for symbol, position in list(self.state.current_positions.items()):
            new_size = position['size'] * 0.5
            await self._reduce_position(symbol, position['size'] - new_size)
        
        # 리스크 이벤트 저장
        await self.db.save_risk_event({
            'event_type': 'restricted',
            'severity': 'high',
            'description': '시장 변동성 심화로 제한 상태 진입',
            'action_taken': '포지션 규모 50% 축소'
        })
            
    async def _handle_emergency_state(self):
        """비상 상태 처리"""
        logging.critical("비상 상태 진입: 모든 거래 중단 및 포지션 청산")
        # 모든 포지션 청산, 거래 중단
        self.state.emergency_mode = True
        await self._close_all_positions()
        
        # 시스템 비상 상태 알림
        await self.db.save_risk_event({
            'event_type': 'emergency',
            'severity': 'critical',
            'description': '시장 변동성 초과로 비상 상태 진입',
            'action_taken': '모든 포지션 청산 및 거래 중단'
        })

    async def _handle_risk_warnings(self, warnings: List[str]):
        """리스크 경고 처리"""
        try:
            for warning in warnings:
                logging.warning(f"Risk warning: {warning}")
            
            self.state.warning_level = 'high' if len(warnings) > 2 else 'medium'
            self.state.active_alerts.extend(warnings)
            
            await self.db.save_risk_event({
                'event_type': 'warning',
                'severity': self.state.warning_level,
                'description': '; '.join(warnings),
                'action_taken': 'Monitoring increased'
            })
            
        except Exception as e:
            logging.error(f"Error handling risk warnings: {e}")

    async def _cleanup_old_data(self):
        """오래된 데이터 정리"""
        try:
            cutoff_time = datetime.now() - timedelta(days=30)
            self.performance_history = [
                data for data in self.performance_history
                if data['timestamp'] > cutoff_time
            ]
            
            self.state.active_alerts = []
            self.last_check = datetime.now()
            logging.info("Completed old data cleanup")
            
        except Exception as e:
            logging.error(f"Error during data cleanup: {e}")

    async def _get_market_data(self) -> Optional[Dict]:
        """시장 데이터 수집"""
        try:
            # 실제 구현에서는 거래소 API 등을 통해 데이터 수집
            return {
                'returns': [0.001, -0.002, 0.003],  # 예시 데이터
                'asset_prices': [[100, 101, 102], [200, 202, 201]],
                'asset_names': ['BTC', 'ETH'],
                'prices': [50000, 50100, 50200, 50150, 50250],  # 기본 가격 데이터
                'prices_1m': [50000, 50100, 50200, 50150, 50250],  # 1분 단위 가격 데이터
                'prices_5m': [49500, 50000, 50500, 51000, 50800],  # 5분 단위 가격 데이터
                'prices_15m': [48000, 49000, 50000, 51000, 52000]  # 15분 단위 가격 데이터
            }
        except Exception as e:
            logging.error(f"Error fetching market data: {e}")
            return None

    def get_risk_status(self) -> Dict:
        """현재 리스크 상태 조회"""
        return {
            'warning_level': self.state.warning_level,
            'risk_level': self.state.risk_level,
            'active_alerts': self.state.active_alerts,
            'current_exposure': self.state.current_exposure,
            'position_count': len(self.state.current_positions),
            'last_update': self.state.last_update,
            'risk_metrics': self.state.risk_metrics,
            'var_levels': self.state.var_levels,
            'emergency_mode': self.state.emergency_mode,
            'timeframe_metrics': self.state.timeframe_metrics
        }

    # ─────────────────────────────────────────────
    # [추가] 포트폴리오 자동 조정 기능 (동적 RL 기반 조정 포함)
    async def _auto_adjust_portfolio(self, risk_metrics: Dict):
        """포트폴리오 비중 자동 조정
           - 시장 변동성이 thresholds.volatility_threshold를 초과하면
             포지션 크기를 비례적으로 축소하며, RL 기반 동적 위험 조정 계수를 곱함.
        """
        try:
            vol = risk_metrics.get('volatility', 0)
            # 기존 정적 factor 계산
            if vol > self.thresholds.volatility_warning and vol > 0:
                factor = self.thresholds.volatility_warning / vol  # factor < 1
            else:
                factor = 1.0
            # RL 기반 동적 조정 계수 산출 (0.5 ~ 1.0 사이)
            dynamic_factor = self._rl_dynamic_adjustment(risk_metrics)
            
            for symbol, pos in self.state.current_positions.items():
                allowed_size = pos['size'] * factor * dynamic_factor
                if pos['size'] > allowed_size:
                    logging.info(f"Auto adjusting position for {symbol}: reducing size from {pos['size']:.4f} to {allowed_size:.4f} (volatility: {vol:.4f}, RL factor: {dynamic_factor:.2f}).")
                    pos['size'] = allowed_size

            self.state.current_exposure = sum(
                pos['size'] * pos.get('entry_price', 0)
                for pos in self.state.current_positions.values()
            )
        except Exception as e:
            logging.error(f"Error in auto-adjusting portfolio: {e}")

    # ─────────────────────────────────────────────
    # [추가] 손실 제한 기능 (자동 청산)
    async def _apply_stop_loss(self, market_data: Dict):
        """포지션별 손실 제한 검사 및 자동 청산
           - 각 포지션의 현재 가격과 진입 가격을 비교하여, 
             손실 비율이 thresholds.max_drawdown 이상이면 해당 포지션을 청산함.
        """
        try:
            symbols_to_liquidate = []
            for symbol, pos in list(self.state.current_positions.items()):
                current_price = self._get_current_price_for_symbol(symbol, market_data)
                if current_price is None:
                    continue
                loss_pct = (pos['entry_price'] - current_price) / pos['entry_price']
                if loss_pct >= self.thresholds.max_drawdown:
                    logging.info(f"Stop loss triggered for {symbol}: loss {loss_pct:.2%} exceeds threshold {self.thresholds.max_drawdown:.2%}. Liquidating position.")
                    symbols_to_liquidate.append(symbol)
            for symbol in symbols_to_liquidate:
                del self.state.current_positions[symbol]
            self.state.current_exposure = sum(
                pos['size'] * pos.get('entry_price', 0)
                for pos in self.state.current_positions.values()
            )
        except Exception as e:
            logging.error(f"Error in applying stop loss: {e}")

    def _get_current_price_for_symbol(self, symbol: str, market_data: Dict) -> Optional[float]:
        """심볼에 해당하는 현재 가격 조회 (시장 데이터 이용)
           - asset_names와 asset_prices를 참고하여 심볼에 포함된 자산의 최신 가격을 반환함.
        """
        try:
            asset_names = market_data.get('asset_names', [])
            asset_prices = market_data.get('asset_prices', [])
            for i, asset in enumerate(asset_names):
                if asset in symbol:
                    if asset_prices and len(asset_prices) > i and asset_prices[i]:
                        return asset_prices[i][-1]
            return None
        except Exception as e:
            logging.error(f"Error in getting current price for {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────
    # [추가] RL 기반 동적 리스크 조정 계수 산출
    def _rl_dynamic_adjustment(self, risk_metrics: Dict) -> float:
        """
        RL 기반 동적 리스크 조정:
         - 현재 리스크 메트릭스(예: volatility)를 입력으로 받아,
           강화학습 모델이 산출한 동적 조정 계수(0.5 ~ 1.0)를 반환한다.
         - (여기서는 모의 구현으로, 변동성이 임계값을 초과할수록 선형적으로 감소하도록 함)
        """
        volatility = risk_metrics.get('volatility', 0)
        if volatility <= self.thresholds.volatility_warning:
            return 1.0
        else:
            scaling = 10.0  # 임의의 스케일링 계수
            factor = 1.0 - (volatility - self.thresholds.volatility_warning) * scaling
            return max(0.5, factor)

    async def _save_risk_state(self):
        """현재 리스크 상태를 DB에 저장"""
        try:
            await self.db.save_risk_state({
                'timestamp': datetime.now(),
                'exposure': self.state.current_exposure,
                'positions': self.state.current_positions,
                'risk_metrics': self.state.risk_metrics,
                'var_levels': self.state.var_levels,
                'warning_level': self.state.warning_level,
                'risk_level': self.state.risk_level,
                'active_alerts': self.state.active_alerts,
                'timeframe_metrics': self.state.timeframe_metrics
            })
        except Exception as e:
            logging.error(f"Error saving risk state: {e}")

    async def _update_risk_metrics(self):
        """포지션 업데이트 후 리스크 메트릭스 갱신 (필요 시 추가 구현)"""
        try:
            # 임시 구현: 현재 익스포저 재계산
            self.state.current_exposure = sum(
                pos['size'] * pos.get('entry_price', 0)
                for pos in self.state.current_positions.values()
            )
        except Exception as e:
            logging.error(f"Error updating risk metrics: {e}")

    def _check_position_limits(self, trade_data: Dict) -> Dict:
        """포지션 크기 한도 체크 (기존 로직)"""
        symbol = trade_data.get('symbol', '')
        current_size = self.state.current_positions.get(symbol, {}).get('size', 0)
        allowed_size = self.thresholds.max_position_size - current_size
        is_safe = allowed_size > 0
        return {'is_safe': is_safe, 'allowed_size': allowed_size, 'reason': '' if is_safe else 'Max position size exceeded'}

    def _check_var_limits(self, trade_data: Dict) -> Dict:
        """VaR 한도 체크 (기존 로직)"""
        # 임시 구현: 항상 안전하다고 가정
        return {'is_safe': True, 'allowed_size': self.thresholds.max_position_size, 'reason': ''}

    async def _check_liquidity(self, trade_data: Dict) -> Dict:
        """유동성 체크 (기존 로직)"""
        # 임시 구현: 항상 안전하다고 가정
        return {'is_safe': True, 'allowed_size': self.thresholds.max_position_size, 'reason': ''}
    
    def _check_emergency_conditions(self, risk_metrics: Dict) -> bool:
        """긴급 상황 조건 체크 (기존 로직)"""
        # 임시 구현: 변동성이 매우 높으면 긴급 상황으로 간주
        return risk_metrics.get('volatility', 0) > (self.thresholds.volatility_threshold * 2)
    
    async def _handle_emergency_situation(self):
        """긴급 상황 처리 (기존 로직)"""
        logging.warning("Emergency conditions met! Taking emergency actions.")
        self.state.emergency_mode = True
        # 추가 조치: 모든 포지션 청산 등 (실제 거래소 API 호출 필요)
        self.state.current_positions.clear()
        self.state.current_exposure = 0
        await self.db.save_risk_event({
            'event_type': 'emergency',
            'severity': 'critical',
            'description': 'Emergency liquidation executed due to extreme market conditions.',
            'action_taken': 'All positions liquidated'
        })

    async def _reduce_position(self, symbol: str, reduction_size: float):
        """포지션 크기 감소"""
        try:
            if symbol not in self.state.current_positions:
                logging.warning(f"Cannot reduce position for {symbol}: position not found")
                return
                
            position = self.state.current_positions[symbol]
            if position['size'] <= reduction_size:
                await self._close_position(symbol, "full_reduction")
                return

            # 포지션 크기 업데이트
            position['size'] -= reduction_size
            logging.info(f"Reduced position for {symbol} by {reduction_size}")

        except Exception as e:
            logging.error(f"Position reduction error: {e}")

    async def _close_position(self, symbol: str, reason: str):
        """포지션 청산"""
        # 여기에 청산 로직 구현 (실제 거래 실행)
        if symbol in self.state.current_positions:
            logging.info(f"Closing position for {symbol} due to {reason}")
            del self.state.current_positions[symbol]
            return True
        return False

    async def _close_all_positions(self):
        """모든 포지션 청산"""
        for symbol in list(self.state.current_positions.keys()):
            await self._close_position(symbol, "emergency_closure")
        self.state.current_positions.clear()

if __name__ == "__main__":
    # 테스트 코드
    async def test_risk_manager():
        logging.basicConfig(level=logging.INFO)
        
        config = {
            'max_position_size': 0.1,
            'var_limit': 0.02,
            'volatility_warning': 0.05,
            'volatility_restricted': 0.08,
            'volatility_emergency': 0.12
        }
        
        risk_manager = ImprovedRiskManager(config)
        
        try:
            # 리스크 모니터링 시작
            monitoring_task = asyncio.create_task(risk_manager.start_monitoring())
            
            # 테스트 거래 데이터
            trade_data = {
                'symbol': 'BTC-KRW',
                'size': 0.1,
                'price': 50000000
            }
            
            # 리스크 체크
            result = await risk_manager.check_trade(trade_data)
            logging.info(f"Risk check result: {result}")
            
            # 포지션 업데이트
            await risk_manager.update_position(trade_data)
            
            # 잠시 대기 후 현재 리스크 상태 출력
            await asyncio.sleep(2)
            status = risk_manager.get_risk_status()
            logging.info(f"Current risk status: {status}")
            
            # 모니터링 종료를 위해 running 플래그 해제
            risk_manager.running = False
            await monitoring_task
            
        except Exception as e:
            logging.error(f"Test failed: {e}")
        finally:
            risk_manager.running = False

    asyncio.run(test_risk_manager())