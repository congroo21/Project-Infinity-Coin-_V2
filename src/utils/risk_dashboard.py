# src/utils/risk_dashboard.py

import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from src.utils.risk_utils import risk_calculator
from src.exceptions import ValidationError

class RiskDashboard:
    """통합 리스크 대시보드"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_update = None
        self.metrics = {}
        self.risk_levels = {
            'low': {'color': 'green', 'threshold': 0.25},
            'medium': {'color': 'yellow', 'threshold': 0.5},
            'high': {'color': 'orange', 'threshold': 0.75},
            'critical': {'color': 'red', 'threshold': 1.0}
        }
    
    def update_metrics(self, sources: Dict[str, Dict]):
        """
        여러 소스에서 리스크 메트릭스 업데이트
        
        Args:
            sources: 소스별 리스크 메트릭스 딕셔너리
                     {'market_analyzer': {...}, 'risk_manager': {...}, ...}
        """
        try:
            for source, metrics in sources.items():
                if metrics:
                    # 기존 소스 정보가 있으면 업데이트, 없으면 새로 추가
                    if source in self.metrics:
                        self.metrics[source].update(metrics)
                    else:
                        self.metrics[source] = metrics

            self.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"메트릭스 업데이트 오류: {e}")
    
    def get_consolidated_metrics(self) -> Dict:
        """
        통합 리스크 메트릭스 반환
        
        Returns:
            통합 리스크 메트릭스 딕셔너리
        """
        try:
            if not self.metrics:
                return {}
                
            # 핵심 지표를 각 소스에서 추출하여 통합
            volatility = self._get_best_metric('volatility')
            var_95 = self._get_best_metric('parametric_var_95')
            sharpe = self._get_best_metric('sharpe_ratio')
            max_dd = self._get_best_metric('max_drawdown')
            
            # 종합 리스크 점수 계산
            risk_score = self._calculate_risk_score({
                'volatility': volatility,
                'parametric_var_95': var_95,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd
            })
            
            # 리스크 수준 결정
            risk_level = self._determine_risk_level(risk_score)
            
            return {
                'timestamp': self.last_update,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'key_metrics': {
                    'volatility': volatility,
                    'var_95': var_95,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd
                },
                'source_metrics': self.metrics
            }
            
        except Exception as e:
            self.logger.error(f"통합 메트릭스 계산 오류: {e}")
            return {}
    
    def _get_best_metric(self, metric_name: str) -> float:
        """
        여러 소스에서 가장 신뢰할 수 있는 메트릭스 값 선택
        
        Args:
            metric_name: 메트릭스 이름
            
        Returns:
            선택된 메트릭스 값
        """
        try:
            values = []
            
            # 각 소스에서 메트릭스 값 수집
            for source, metrics in self.metrics.items():
                if metric_name in metrics:
                    values.append(metrics[metric_name])
            
            if not values:
                return 0.0
                
            # 중간값 반환 (극단값 영향 최소화)
            return float(np.median(values))
            
        except Exception as e:
            self.logger.error(f"최적 메트릭스 선택 오류: {e}")
            return 0.0
    
    def _calculate_risk_score(self, metrics: Dict[str, float]) -> float:
        """
        종합 리스크 점수 계산
        
        Args:
            metrics: 주요 메트릭스 딕셔너리
            
        Returns:
            0에서 1 사이의 리스크 점수
        """
        try:
            # 각 메트릭스를 0-1 범위로 정규화
            normalized = {}
            
            # 변동성 (높을수록 위험)
            normalized['volatility'] = min(1.0, metrics['volatility'] / 0.05)
            
            # VaR (높을수록 위험)
            normalized['var'] = min(1.0, metrics['parametric_var_95'] / 0.03)
            
            # 샤프 비율 (낮을수록 위험)
            sharpe = metrics['sharpe_ratio']
            normalized['sharpe'] = max(0.0, min(1.0, (2.0 - sharpe) / 2.0)) if sharpe > 0 else 1.0
            
            # 최대 낙폭 (높을수록 위험)
            normalized['drawdown'] = min(1.0, metrics['max_drawdown'] / 0.2)
            
            # 가중 평균 계산
            weights = {'volatility': 0.3, 'var': 0.3, 'sharpe': 0.2, 'drawdown': 0.2}
            
            risk_score = sum(normalized[k] * weights[k] for k in weights)
            
            return min(1.0, max(0.0, risk_score))
            
        except Exception as e:
            self.logger.error(f"리스크 점수 계산 오류: {e}")
            return 0.5
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """
        리스크 수준 결정
        
        Args:
            risk_score: 리스크 점수 (0-1)
            
        Returns:
            리스크 수준 문자열 ('low', 'medium', 'high', 'critical')
        """
        try:
            for level, info in sorted(self.risk_levels.items(), key=lambda x: x[1]['threshold']):
                if risk_score <= info['threshold']:
                    return level
            return 'critical'
            
        except Exception as e:
            self.logger.error(f"리스크 수준 결정 오류: {e}")
            return 'medium'

# 싱글톤 인스턴스
risk_dashboard = RiskDashboard()