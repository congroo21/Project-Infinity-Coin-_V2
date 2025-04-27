# src/scenarios/monitoring/performance.py

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from collections import deque
import psutil

from ...exceptions import SystemResourceError, ValidationError

@dataclass
class PerformanceMetrics:
    """성능 지표 데이터 클래스"""
    timestamp: datetime
    total_trades: int
    successful_trades: int
    total_pnl: float
    win_rate: float
    average_latency: float
    cpu_usage: float
    memory_usage: float
    errors_count: int
    throughput: float  # 초당 처리량

class ScenarioPerformanceMonitor:
    """향상된 성능 모니터링 시스템"""
    def __init__(self):
        # 시작 시간 및 카운터
        self.start_time = datetime.now()
        self.total_messages = 0
        self.error_count = 0
        self.successful_trades = 0
        
        # 성능 이력
        self.latency_history = deque(maxlen=1000)  # 지연시간 이력
        self.pnl_history = deque(maxlen=1000)      # 손익 이력
        self.cpu_history = deque(maxlen=100)       # CPU 사용량 이력
        self.memory_history = deque(maxlen=100)    # 메모리 사용량 이력
        self.throughput_history = deque(maxlen=60)  # 처리량 이력 (1분)
        
        # 실행 상태
        self.running = False
        self.critical_error = False
        self.last_update = None
        
        # 임계값 설정
        self.thresholds = {
            'cpu_critical': 90,       # CPU 90% 이상
            'memory_critical': 85,    # 메모리 85% 이상
            'latency_critical': 1.0,  # 1초 이상 지연
            'error_rate_critical': 0.1  # 10% 이상 오류율

        }
        
         # 초기화시 execution_stats 추가
        self.execution_stats = {
            'execution_time': 0.0,
            'success_rate': 0.0,
            'error_rate': 0.0,
            'avg_latency': 0.0
        }
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)

    async def start_monitoring(self):
        """모니터링 시작"""
        self.running = True
        self.logger.info("Performance monitoring started")
        
        try:
            monitor_task = asyncio.create_task(self._monitoring_loop())
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            await asyncio.gather(monitor_task, cleanup_task)
            
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            self.running = False
            raise

    async def _monitoring_loop(self):
        """메인 모니터링 루프"""
        while self.running:
            try:
                # 시스템 리소스 모니터링
                await self._monitor_system_resources()
                
                # 성능 지표 계산
                metrics = self._calculate_performance_metrics()
                
                # 임계값 체크
                if self._check_thresholds(metrics):
                    await self._handle_critical_situation(metrics)
                
                # 지표 기록
                self._record_metrics(metrics)
                
                # 모니터링 간격 (100ms)
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1)

    async def _cleanup_loop(self):
        """데이터 정리 루프"""
        while self.running:
            try:
                # 오래된 데이터 정리
                self._cleanup_old_data()
                
                # 1시간마다 실행
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)

    def record_trade(self, success: bool, pnl: float, latency: float):
        """거래 결과 기록"""
        try:
            self.total_messages += 1
            if success:
                self.successful_trades += 1
                self.pnl_history.append(pnl)
                self.latency_history.append(latency)
            else:
                self.error_count += 1
            
            self.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Trade recording error: {e}")

    def record_error(self, error_type: str, message: str):
        """오류 기록"""
        try:
            self.error_count += 1
            self.logger.error(f"{error_type}: {message}")
            
            if self.error_count / max(1, self.total_messages) > self.thresholds['error_rate_critical']:
                self.critical_error = True
                self.logger.critical("Critical error rate exceeded!")
                
        except Exception as e:
            self.logger.error(f"Error recording error: {e}")

    async def _monitor_system_resources(self):
        """시스템 리소스 모니터링"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            self.cpu_history.append(cpu_percent)
            self.memory_history.append(memory_percent)
            
            # 임계값 체크
            if (cpu_percent > self.thresholds['cpu_critical'] or
                memory_percent > self.thresholds['memory_critical']):
                raise SystemResourceError(
                    f"Resource limits exceeded: CPU {cpu_percent}%, Memory {memory_percent}%"
                )
                
        except Exception as e:
            self.logger.error(f"Resource monitoring error: {e}")

    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """성능 지표 계산"""
        try:
            # 실행 시간 계산
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # 승률 계산
            win_rate = self.successful_trades / max(1, self.total_messages)
            
            # 평균 지연시간
            avg_latency = (np.mean(self.latency_history)
                          if self.latency_history else 0.0)
            
            # 처리량 계산 (초당)
            throughput = self.total_messages / max(1, uptime)
            
            # CPU/메모리 사용량
            cpu_usage = np.mean(self.cpu_history) if self.cpu_history else 0.0
            memory_usage = np.mean(self.memory_history) if self.memory_history else 0.0
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                total_trades=self.total_messages,
                successful_trades=self.successful_trades,
                total_pnl=sum(self.pnl_history),
                win_rate=win_rate * 100,
                average_latency=avg_latency,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                errors_count=self.error_count,
                throughput=throughput
            )
            
        except Exception as e:
            self.logger.error(f"Metrics calculation error: {e}")
            return None

    def _check_thresholds(self, metrics: PerformanceMetrics) -> bool:
        """임계값 체크"""
        try:
            if not metrics:
                return False
                
            return any([
                metrics.cpu_usage > self.thresholds['cpu_critical'],
                metrics.memory_usage > self.thresholds['memory_critical'],
                metrics.average_latency > self.thresholds['latency_critical'],
                (metrics.errors_count / max(1, metrics.total_trades)) > 
                    self.thresholds['error_rate_critical']
            ])
            
        except Exception as e:
            self.logger.error(f"Threshold check error: {e}")
            return False

    async def _handle_critical_situation(self, metrics: PerformanceMetrics):
        """임계 상황 처리"""
        try:
            self.critical_error = True
            
            # 상황 로깅
            self.logger.critical(
                "Critical situation detected:\n"
                f"CPU Usage: {metrics.cpu_usage:.1f}%\n"
                f"Memory Usage: {metrics.memory_usage:.1f}%\n"
                f"Average Latency: {metrics.average_latency:.3f}s\n"
                f"Error Rate: {(metrics.errors_count / max(1, metrics.total_trades)) * 100:.1f}%"
            )
            
            # 알림 발송 (실제 구현 필요)
            await self._send_alert(metrics)
            
        except Exception as e:
            self.logger.error(f"Critical situation handling error: {e}")

    def _cleanup_old_data(self):
        """오래된 데이터 정리"""
        try:
            # 24시간 이상된 데이터 정리
            cutoff_time = datetime.now() - timedelta(days=1)
            
            # 이력 데이터 정리 로직 구현
            # (실제 구현 시 필요에 따라 구체화)
            
        except Exception as e:
            self.logger.error(f"Data cleanup error: {e}")

    async def _send_alert(self, metrics: PerformanceMetrics):
        """알림 발송"""
        # 실제 알림 발송 로직 구현 필요
        pass

    def get_performance_summary(self) -> Dict:
        """성능 요약 정보"""
        try:
            metrics = self._calculate_performance_metrics()
            if not metrics:
                return {}
            
            return {
                'timestamp': metrics.timestamp.isoformat(),
                'execution_stats': {
                    'total_trades': metrics.total_trades,
                    'successful_trades': metrics.successful_trades,
                    'win_rate': f"{metrics.win_rate:.1f}%",
                    'total_pnl': f"{metrics.total_pnl:.2f}",
                    'average_latency': f"{metrics.average_latency*1000:.2f}ms"
                },
                'system_stats': {
                    'cpu_usage': f"{metrics.cpu_usage:.1f}%",
                    'memory_usage': f"{metrics.memory_usage:.1f}%",
                    'error_count': metrics.errors_count,
                    'throughput': f"{metrics.throughput:.1f}/s"
                },
                'status': {
                    'running': self.running,
                    'critical_error': self.critical_error,
                    'last_update': (self.last_update.isoformat() 
                                  if self.last_update else None)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
            return {}
    def get_performance_metrics(self) -> Dict:
        """성능 메트릭스 조회"""
        try:
            return {
                'execution_time': self.execution_stats.get('execution_time', 0.0),
                'success_rate': self.execution_stats.get('success_rate', 0.0),
                'error_rate': self.execution_stats.get('error_rate', 0.0),
                'avg_latency': self.execution_stats.get('avg_latency', 0.0)
            }
        except Exception as e:
            logging.error(f"성능 메트릭스 조회 오류: {e}")
            return {}