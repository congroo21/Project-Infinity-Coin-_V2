# src/scenarios/models/bayesian_model.py

import logging
from datetime import datetime
from typing import Dict, Tuple, List

class BayesianModel:
    """베이지안 확률 모델"""
    def __init__(self):
        # 시나리오별 사전 확률 설정
        self.scenario_priors = {
            'high_volatility': (2.0, 3.0),  # (alpha, beta)
            'low_volatility': (3.0, 2.0),
            'strong_trend': (3.0, 2.0),
            'weak_trend': (2.0, 2.0),
            'high_liquidity': (3.0, 2.0),
            'low_liquidity': (2.0, 3.0)
        }
        
        self.min_data_points = 10
        self.history = []

    def update(self, scenario_type: str, success: bool, return_rate: float):
        """시나리오별 베이지안 업데이트"""
        try:
            # 거래 결과 기록
            self.history.append({
                'timestamp': datetime.now(),
                'scenario_type': scenario_type,
                'success': success,
                'return_rate': return_rate
            })
            
            # 해당 시나리오의 사전 확률 업데이트
            if scenario_type in self.scenario_priors:
                alpha, beta = self.scenario_priors[scenario_type]
                
                if success:
                    # 성공 시 alpha 증가
                    self.scenario_priors[scenario_type] = (alpha + 1, beta)
                else:
                    # 실패 시 beta 증가
                    self.scenario_priors[scenario_type] = (alpha, beta + 1)

        except Exception as e:
            logging.error(f"베이지안 모델 업데이트 오류: {e}")

    def get_scenario_probability(self, scenario_type: str) -> Tuple[float, float]:
        """시나리오별 확률과 신뢰도 계산"""
        try:
            if scenario_type not in self.scenario_priors:
                return (0.5, 0.0)  # 기본값
                
            alpha, beta = self.scenario_priors[scenario_type]
            
            # 성공 확률 계산 (alpha / (alpha + beta))
            probability = alpha / (alpha + beta)
            
            # 신뢰도는 데이터 포인트가 많을수록 높아짐
            total_points = alpha + beta - 4.0  # 초기값(4.0) 제외
            confidence = min(1.0, total_points / self.min_data_points)
            
            return (probability, confidence)

        except Exception as e:
            logging.error(f"확률 계산 오류: {e}")
            return (0.5, 0.0)

    def get_scenario_stats(self, scenario_type: str) -> Dict:
        """시나리오별 통계 정보"""
        try:
            # 해당 시나리오의 거래 기록만 필터링
            scenario_history = [
                record for record in self.history[-100:]  # 최근 100개
                if record['scenario_type'] == scenario_type
            ]
            
            if not scenario_history:
                return {}

            # 승률 계산
            wins = sum(1 for record in scenario_history if record['success'])
            total = len(scenario_history)
            win_rate = wins / total

            # 평균 수익률 계산
            avg_return = sum(
                record['return_rate'] for record in scenario_history
            ) / total

            # 확률과 신뢰도 가져오기
            probability, confidence = self.get_scenario_probability(scenario_type)

            return {
                'win_rate': win_rate,
                'average_return': avg_return,
                'probability': probability,
                'confidence': confidence,
                'total_trades': total
            }

        except Exception as e:
            logging.error(f"시나리오 통계 계산 오류: {e}")
            return {}

    def get_best_scenarios(self) -> List[Tuple[str, float]]:
        """현재 가장 높은 확률을 가진 시나리오 목록"""
        try:
            scenario_probs = []
            for scenario_type in self.scenario_priors:
                prob, conf = self.get_scenario_probability(scenario_type)
                # 확률과 신뢰도를 곱하여 최종 점수 계산
                final_score = prob * conf
                scenario_probs.append((scenario_type, final_score))

            # 점수 기준 내림차순 정렬
            return sorted(
                scenario_probs,
                key=lambda x: x[1],
                reverse=True
            )

        except Exception as e:
            logging.error(f"베스트 시나리오 계산 오류: {e}")
            return []