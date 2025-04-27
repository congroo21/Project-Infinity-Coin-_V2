# src/testing/ab_testing.py

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from scipy import stats

@dataclass
class ExperimentConfig:
    """실험 설정"""
    experiment_id: str
    duration_seconds: int = 3600  # 1시간
    control_weight: float = 0.5   # 기존 전략 비중
    test_weight: float = 0.5      # 새로운 전략 비중
    min_sample_size: int = 100    # 최소 샘플 수

@dataclass
class ExperimentResult:
    """실험 결과"""
    experiment_id: str
    start_time: datetime
    end_time: datetime
    control_metrics: Dict
    test_metrics: Dict
    winner: Optional[str] = None
    p_value: float = 1.0
    improvement: float = 0.0

class TradingABTest:
    """트레이딩 전략 A/B 테스트"""
    def __init__(self):
        self.active_experiments = {}
        self.completed_experiments = []
        self.current_experiment = None

    async def start_experiment(self, config: ExperimentConfig, control_strategy: Dict, test_strategy: Dict):
        """새로운 실험 시작"""
        if config.experiment_id in self.active_experiments:
            raise ValueError(f"실험 ID가 이미 존재함: {config.experiment_id}")

        experiment = {
            'config': config,
            'start_time': datetime.now(),
            'control_strategy': control_strategy,
            'test_strategy': test_strategy,
            'control_results': [],
            'test_results': []
        }

        self.active_experiments[config.experiment_id] = experiment
        self.current_experiment = experiment

        # 실험 종료 스케줄링
        asyncio.create_task(self._end_experiment(config.experiment_id, config.duration_seconds))

        logging.info(f"실험 시작: {config.experiment_id}")
        return experiment

    async def _end_experiment(self, experiment_id: str, duration: int):
        """실험 종료 처리"""
        await asyncio.sleep(duration)
        
        if experiment_id in self.active_experiments:
            experiment = self.active_experiments[experiment_id]
            result = self._analyze_results(experiment)
            
            self.completed_experiments.append(result)
            del self.active_experiments[experiment_id]
            
            if self.current_experiment and self.current_experiment['config'].experiment_id == experiment_id:
                self.current_experiment = None

            logging.info(f"실험 종료: {experiment_id}, 승자: {result.winner}")
            return result

    def record_trade_result(self, trade_result: Dict):
        """거래 결과 기록"""
        if not self.current_experiment:
            return

        # 확률에 따라 control 또는 test 그룹에 할당
        if np.random.random() < self.current_experiment['config'].control_weight:
            self.current_experiment['control_results'].append(trade_result)
        else:
            self.current_experiment['test_results'].append(trade_result)

    def _analyze_results(self, experiment: Dict) -> ExperimentResult:
        """실험 결과 분석"""
        control_profits = [r['profit'] for r in experiment['control_results']]
        test_profits = [r['profit'] for r in experiment['test_results']]

        if len(control_profits) < experiment['config'].min_sample_size or \
           len(test_profits) < experiment['config'].min_sample_size:
            return ExperimentResult(
                experiment_id=experiment['config'].experiment_id,
                start_time=experiment['start_time'],
                end_time=datetime.now(),
                control_metrics=self._calculate_metrics(control_profits),
                test_metrics=self._calculate_metrics(test_profits)
            )

        # t-검정 수행
        t_stat, p_value = stats.ttest_ind(control_profits, test_profits)
        
        # 승자 결정 (p < 0.05인 경우에만)
        winner = None
        improvement = 0.0
        if p_value < 0.05:
            control_mean = np.mean(control_profits)
            test_mean = np.mean(test_profits)
            if test_mean > control_mean:
                winner = 'test'
                improvement = (test_mean - control_mean) / abs(control_mean) * 100
            else:
                winner = 'control'
                improvement = (control_mean - test_mean) / abs(test_mean) * 100

        return ExperimentResult(
            experiment_id=experiment['config'].experiment_id,
            start_time=experiment['start_time'],
            end_time=datetime.now(),
            control_metrics=self._calculate_metrics(control_profits),
            test_metrics=self._calculate_metrics(test_profits),
            winner=winner,
            p_value=p_value,
            improvement=improvement
        )

    def _calculate_metrics(self, profits: List[float]) -> Dict:
        """성과 지표 계산"""
        if not profits:
            return {}

        return {
            'total_trades': len(profits),
            'win_rate': len([p for p in profits if p > 0]) / len(profits),
            'avg_profit': np.mean(profits),
            'max_drawdown': self._calculate_max_drawdown(profits),
            'sharpe_ratio': self._calculate_sharpe_ratio(profits)
        }

    def _calculate_max_drawdown(self, profits: List[float]) -> float:
        """최대 손실폭 계산"""
        cumulative = np.cumsum(profits)
        max_so_far = np.maximum.accumulate(cumulative)
        drawdowns = (max_so_far - cumulative) / max_so_far
        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    def _calculate_sharpe_ratio(self, profits: List[float]) -> float:
        """샤프 비율 계산"""
        if len(profits) < 2:
            return 0.0
        return np.mean(profits) / (np.std(profits) + 1e-10) * np.sqrt(252 * 24 * 60) # 연율화

    def get_experiment_stats(self) -> Dict:
        """실험 현황 조회"""
        return {
            'active_experiments': len(self.active_experiments),
            'completed_experiments': len(self.completed_experiments),
            'current_experiment': self.current_experiment['config'].experiment_id if self.current_experiment else None,
            'successful_improvements': len([e for e in self.completed_experiments if e.winner == 'test'])
        }