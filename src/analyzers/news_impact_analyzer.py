import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

@dataclass
class PriceImpactResult:
    """가격 영향 분석 결과"""
    timeframe: str
    price_change: float
    volume_change: float
    correlation: float
    significance: float
    latency: float  # 영향 발생까지 걸린 시간(초)

class NewsImpactAnalyzer:
    """뉴스 영향 분석기"""
    def __init__(self, config: Dict):
        self.config = config
        self.impact_history = deque(maxlen=1000)
        self.correlation_cache = {}
        
        # 분석 설정
        self.min_data_points = 50
        self.significance_threshold = 0.05
        self.correlation_window = 60  # 60초
        
        # 시간프레임 설정
        self.timeframes = [1, 5, 10, 30, 60]  # 초 단위
        
        # 성능 모니터링
        self.processed_count = 0
        self.error_count = 0

    async def analyze_price_impact(
        self,
        news_timestamp: datetime,
        price_data: List[Tuple[datetime, float]],
        volume_data: List[Tuple[datetime, float]]
    ) -> Dict[str, PriceImpactResult]:
        """가격 영향 상세 분석"""
        try:
            results = {}
            base_price = self._get_price_at(price_data, news_timestamp)
            base_volume = self._get_volume_at(volume_data, news_timestamp)
            
            if not base_price or not base_volume:
                return results

            for tf in self.timeframes:
                end_time = news_timestamp + timedelta(seconds=tf)
                price_change = self._calculate_price_change(price_data, news_timestamp, end_time, base_price)
                volume_change = self._calculate_volume_change(volume_data, news_timestamp, end_time, base_volume)
                correlation, significance = self._analyze_correlation(price_data, volume_data, news_timestamp, end_time)
                latency = self._measure_impact_latency(price_data, news_timestamp, end_time, price_change)

                results[f"{tf}s"] = PriceImpactResult(
                    timeframe=f"{tf}s",
                    price_change=price_change,
                    volume_change=volume_change,
                    correlation=correlation,
                    significance=significance,
                    latency=latency
                )
                
            self.impact_history.append((news_timestamp, results))
            self.processed_count += 1
            return results
            
        except Exception as e:
            self.error_count += 1
            logging.error(f"Price impact analysis error: {e}")
            return {}

    def _get_price_at(
        self,
        price_data: List[Tuple[datetime, float]],
        target_time: datetime
    ) -> Optional[float]:
        """특정 시점의 가격 조회"""
        try:
            exact_match = next(
                (price for time, price in price_data if time == target_time),
                None
            )
            if exact_match:
                return exact_match
            previous_prices = [(t, p) for t, p in price_data if t <= target_time]
            if previous_prices:
                return previous_prices[-1][1]
            return None
        except Exception as e:
            logging.error(f"Price lookup error: {e}")
            return None

    def _get_volume_at(
        self,
        volume_data: List[Tuple[datetime, float]],
        target_time: datetime
    ) -> Optional[float]:
        """특정 시점의 거래량 조회"""
        try:
            exact_match = next(
                (vol for time, vol in volume_data if time == target_time),
                None
            )
            if exact_match:
                return exact_match
            previous_volumes = [(t, v) for t, v in volume_data if t <= target_time]
            if previous_volumes:
                return previous_volumes[-1][1]
            return None
        except Exception as e:
            logging.error(f"Volume lookup error: {e}")
            return None

    def _calculate_price_change(
        self,
        price_data: List[Tuple[datetime, float]],
        start_time: datetime,
        end_time: datetime,
        base_price: float
    ) -> float:
        """가격 변화율 계산"""
        try:
            period_prices = [
                p for (t,p) in price_data if start_time <= t <= end_time
            ]
            if not period_prices:
                return 0.0
            end_price = np.mean(period_prices[-5:])  # 마지막 5개 평균
            return (end_price - base_price) / base_price
        except Exception as e:
            logging.error(f"Price change calculation error: {e}")
            return 0.0

    def _calculate_volume_change(
        self,
        volume_data: List[Tuple[datetime, float]],
        start_time: datetime,
        end_time: datetime,
        base_volume: float
    ) -> float:
        """거래량 변화율 계산"""
        try:
            period_volumes = [
                v for (t,v) in volume_data if start_time <= t <= end_time
            ]
            if not period_volumes:
                return 0.0
            avg_volume = np.mean(period_volumes)
            return (avg_volume - base_volume) / base_volume
        except Exception as e:
            logging.error(f"Volume change calculation error: {e}")
            return 0.0

    def _analyze_correlation(
        self,
        price_data: List[Tuple[datetime, float]],
        volume_data: List[Tuple[datetime, float]],
        start_time: datetime,
        end_time: datetime
    ) -> Tuple[float, float]:
        """가격-거래량 상관관계 분석"""
        try:
            period_prices = [p for (t,p) in price_data if start_time <= t <= end_time]
            period_volumes = [v for (t,v) in volume_data if start_time <= t <= end_time]
            if len(period_prices) < 2 or len(period_volumes) < 2:
                return 0.0, 1.0

            correlation = np.corrcoef(period_prices, period_volumes)[0, 1]
            from scipy import stats
            _, p_value = stats.pearsonr(period_prices, period_volumes)
            return correlation, p_value
        except Exception as e:
            logging.error(f"Correlation analysis error: {e}")
            return 0.0, 1.0

    def _measure_impact_latency(
        self,
        price_data: List[Tuple[datetime, float]],
        start_time: datetime,
        end_time: datetime,
        expected_change: float
    ) -> float:
        """가격 영향 발생 지연시간 측정"""
        try:
            if abs(expected_change) < 1e-6:
                return float('inf')
            period_data = [(t,p) for (t,p) in price_data if start_time <= t <= end_time]
            if not period_data:
                return float('inf')
            base_price = period_data[0][1]
            threshold = expected_change * 0.5
            for (t,p) in period_data:
                change = (p - base_price) / base_price
                if abs(change) >= abs(threshold):
                    return (t - start_time).total_seconds()
            return float('inf')
        except Exception as e:
            logging.error(f"Latency measurement error: {e}")
            return float('inf')

    def get_impact_stats(self) -> Dict:
        """영향 분석 통계"""
        if not self.impact_history:
            return {}
        recent_impacts = list(self.impact_history)[-100:]

        stats = {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'average_latency': {},
            'average_correlation': {},
            'significance_ratio': {}
        }
        
        for tf in self.timeframes:
            tf_key = f"{tf}s"
            tf_impacts = [
                impact[1][tf_key] for impact in recent_impacts if tf_key in impact[1]
            ]
            if tf_impacts:
                latencies = [imp.latency for imp in tf_impacts if imp.latency != float('inf')]
                stats['average_latency'][tf_key] = np.mean(latencies) if latencies else float('inf')
                stats['average_correlation'][tf_key] = np.mean([imp.correlation for imp in tf_impacts])
                stats['significance_ratio'][tf_key] = np.mean([
                    1 if imp.significance < self.significance_threshold else 0
                    for imp in tf_impacts
                ])
        return stats
