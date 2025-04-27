# src/performance_monitor.py

import psutil
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
import numpy as np
from collections import deque

from .exceptions import SystemResourceError, ValidationError

@dataclass
class SystemMetrics:
    """시스템 메트릭스 데이터 클래스"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]  # 'sent', 'received' bytes
    process_count: int

@dataclass
class PerformanceStats:
    """성능 통계 데이터 클래스"""
    elapsed_time: float
    total_messages: int
    messages_per_second: float
    error_count: int
    latency_avg: float
    latency_max: float

class PerformanceMonitor:
    """기본 성능 모니터링 시스템"""
    def __init__(self):
        # 시작 시간 및 카운터
        self.start_time = time.time()
        self.total_messages = 0
        self.error_count = 0
        
        # 성능 측정 데이터
        self.latency_history = deque(maxlen=1000)  # 최근 1000개 지연시간
        self.message_timestamps = deque(maxlen=100)  # 처리량 계산용
        self.system_metrics = deque(maxlen=60)  # 1분치 시스템 메트릭스
        
        # 실행 상태
        self.running = False
        self.monitor_task = None
        
        # 임계값 설정
        self.thresholds = {
            'cpu_limit': 80,        # 80% CPU
            'memory_limit': 85,     # 85% Memory
            'disk_limit': 90,       # 90% Disk
            'latency_limit': 0.1    # 100ms
        }
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """로깅 설정"""
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    async def start(self):
        """모니터링 시작"""
        if self.running:
            return
            
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Performance monitoring started")

    async def stop(self):
        """모니터링 중지"""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self):
        """메인 모니터링 루프"""
        while self.running:
            try:
                # 시스템 메트릭스 수집
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                
                # 임계값 체크
                await self._check_thresholds(metrics)
                
                # 처리량 계산 및 로깅
                stats = self.get_stats()
                if self.total_messages % 1000 == 0:  # 1000개 메시지마다 로깅
                    self.log_stats()
                
                await asyncio.sleep(1)  # 1초 간격 모니터링
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)

    def record_message(self, latency: Optional[float] = None):
        """메시지 처리 기록"""
        try:
            self.total_messages += 1
            now = time.time()
            
            self.message_timestamps.append(now)
            
            if latency is not None:
                self.latency_history.append(latency)
                
                if latency > self.thresholds['latency_limit']:
                    self.logger.warning(
                        f"High latency detected: {latency*1000:.2f}ms"
                    )
                    
        except Exception as e:
            self.logger.error(f"Message recording error: {e}")

    def record_error(self):
        """오류 기록"""
        self.error_count += 1

    def _collect_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭스 수집"""
        try:
            network_io = psutil.net_io_counters()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                disk_usage=psutil.disk_usage('/').percent,
                network_io={
                    'sent': network_io.bytes_sent,
                    'received': network_io.bytes_recv
                },
                process_count=len(psutil.pids())
            )
            
        except Exception as e:
            self.logger.error(f"Metrics collection error: {e}")
            return None

    async def _check_thresholds(self, metrics: SystemMetrics):
        """임계값 체크"""
        if not metrics:
            return
            
        try:
            violations = []
            
            if metrics.cpu_usage > self.thresholds['cpu_limit']:
                violations.append(f"CPU usage: {metrics.cpu_usage:.1f}%")
                
            if metrics.memory_usage > self.thresholds['memory_limit']:
                violations.append(f"Memory usage: {metrics.memory_usage:.1f}%")
                
            if metrics.disk_usage > self.thresholds['disk_limit']:
                violations.append(f"Disk usage: {metrics.disk_usage:.1f}%")
                
            if violations:
                message = "Resource limits exceeded: " + ", ".join(violations)
                self.logger.warning(message)
                await self._handle_threshold_violation(message)
                
        except Exception as e:
            self.logger.error(f"Threshold check error: {e}")

    async def _handle_threshold_violation(self, message: str):
        """임계값 위반 처리"""
        try:
            # 시스템 상태 로깅
            self.logger.warning(f"System status: {self.get_system_status()}")
            
            # 알림 발송 (구현 필요)
            await self._send_alert(message)
            
        except Exception as e:
            self.logger.error(f"Violation handling error: {e}")

    def get_stats(self) -> PerformanceStats:
        """성능 통계 계산"""
        try:
            elapsed_time = time.time() - self.start_time
            messages_per_second = self.total_messages / max(1, elapsed_time)
            
            latency_stats = self._calculate_latency_stats()
            
            return PerformanceStats(
                elapsed_time=elapsed_time,
                total_messages=self.total_messages,
                messages_per_second=messages_per_second,
                error_count=self.error_count,
                latency_avg=latency_stats['avg'],
                latency_max=latency_stats['max']
            )
            
        except Exception as e:
            self.logger.error(f"Stats calculation error: {e}")
            return None

    def _calculate_latency_stats(self) -> Dict[str, float]:
        """지연시간 통계 계산"""
        try:
            if not self.latency_history:
                return {'avg': 0.0, 'max': 0.0}
                
            latencies = list(self.latency_history)
            return {
                'avg': float(np.mean(latencies)),
                'max': float(np.max(latencies))
            }
            
        except Exception as e:
            self.logger.error(f"Latency calculation error: {e}")
            return {'avg': 0.0, 'max': 0.0}

    def get_system_status(self) -> Dict:
        """시스템 상태 정보"""
        try:
            if not self.system_metrics:
                return {}
                
            latest = self.system_metrics[-1]
            return {
                'timestamp': latest.timestamp.isoformat(),
                'cpu_usage': f"{latest.cpu_usage:.1f}%",
                'memory_usage': f"{latest.memory_usage:.1f}%",
                'disk_usage': f"{latest.disk_usage:.1f}%",
                'network_io': {
                    'sent': self._format_bytes(latest.network_io['sent']),
                    'received': self._format_bytes(latest.network_io['received'])
                },
                'process_count': latest.process_count
            }
            
        except Exception as e:
            self.logger.error(f"Status retrieval error: {e}")
            return {}

    def log_stats(self):
        """상태 정보 로깅"""
        try:
            stats = self.get_stats()
            if not stats:
                return
                
            self.logger.info(
                f"💫 실행 시간: {stats.elapsed_time:.1f}s | "
                f"📊 총 메시지: {stats.total_messages} | "
                f"⚡ 처리율: {stats.messages_per_second:.1f}/s | "
                f"❌ 오류: {stats.error_count} | "
                f"⏱ 평균 지연: {stats.latency_avg*1000:.2f}ms | "
                f"⚠️ 최대 지연: {stats.latency_max*1000:.2f}ms"
            )
            
        except Exception as e:
            self.logger.error(f"Stats logging error: {e}")

    @staticmethod
    def _format_bytes(bytes: int) -> str:
        """바이트 단위 포맷팅"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024:
                return f"{bytes:.1f}{unit}"
            bytes /= 1024
        return f"{bytes:.1f}TB"

    async def _send_alert(self, message: str):
        """알림 발송"""
        # 실제 알림 발송 로직 구현 필요
        pass

# 테스트 코드
async def run_performance_test():
    monitor = PerformanceMonitor()
    await monitor.start()
    
    try:
        # 테스트 데이터 생성
        for i in range(10000):
            monitor.record_message(latency=0.001)  # 1ms 지연
            if i % 500 == 0:
                await asyncio.sleep(0.001)
        
        monitor.record_error()  # 오류 테스트
        await asyncio.sleep(2)
        
    finally:
        await monitor.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_performance_test())