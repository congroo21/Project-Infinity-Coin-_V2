import psutil
import logging
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
import asyncio
from collections import deque

@dataclass
class MemoryConfig:
    """메모리 관리 설정"""
    warning_threshold_mb: int = 1000  # 경고 임계값 (MB)
    critical_threshold_mb: int = 1500  # 위험 임계값 (MB)
    check_interval_ms: int = 100      # 체크 주기 (밀리초)
    cleanup_threshold_mb: int = 1200   # 정리 시작 임계값 (MB)
    buffer_reduce_percent: int = 50    # 버퍼 감소율 (%)

class MemoryManager:
    """메모리 사용량 관리"""
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.process = psutil.Process()
        self.usage_history = deque(maxlen=1000)  # 최근 1000개 기록 유지
        self.last_cleanup_time = datetime.now()
        self.running = False
        self.warning_callbacks = []
        self.critical_callbacks = []

    async def start_monitoring(self):
        """메모리 모니터링 시작"""
        self.running = True
        while self.running:
            try:
                memory_info = self._check_memory()
                self.usage_history.append(memory_info)

                # 임계값 체크 및 조치
                await self._handle_memory_state(memory_info)
                
                # 체크 주기만큼 대기
                await asyncio.sleep(self.config.check_interval_ms / 1000)
                
            except Exception as e:
                logging.error(f"메모리 모니터링 오류: {e}")
                await asyncio.sleep(1)  # 오류 발생 시 1초 대기

    def stop_monitoring(self):
        """모니터링 중지"""
        self.running = False

    def _check_memory(self) -> Dict:
        """현재 메모리 사용량 체크"""
        mem_info = self.process.memory_info()
        system_mem = psutil.virtual_memory()
        
        return {
            'timestamp': datetime.now(),
            'rss': mem_info.rss / (1024 * 1024),  # MB 단위
            'vms': mem_info.vms / (1024 * 1024),
            'system_used_percent': system_mem.percent
        }

    async def _handle_memory_state(self, memory_info: Dict):
        """메모리 상태에 따른 처리 - 최적화 버전"""
        current_usage_mb = memory_info['rss']
        
        # 메모리 사용량에 따른 처리 주기 조정
        if current_usage_mb >= self.config.critical_threshold_mb:
            # 위험 상태 (즉시 처리)
            await self._handle_critical_state(current_usage_mb)
        elif current_usage_mb >= self.config.warning_threshold_mb:
            # 경고 상태 (10초마다 처리)
            if (datetime.now() - self._last_warning_handle).total_seconds() > 10:
                await self._handle_warning_state(current_usage_mb)
                self._last_warning_handle = datetime.now()
        elif current_usage_mb >= self.config.cleanup_threshold_mb:
            # 정리 필요 상태 (30초마다 처리)
            if (datetime.now() - self._last_cleanup).total_seconds() > 30:
                await self._handle_cleanup_state(current_usage_mb)
                self._last_cleanup = datetime.now()

    async def _handle_critical_state(self, usage_mb: float):
        """위험 상태 처리"""
        logging.critical(f"메모리 사용량 위험 수준: {usage_mb:.1f}MB")
        
        # 등록된 콜백 실행
        for callback in self.critical_callbacks:
            try:
                await callback(usage_mb)
            except Exception as e:
                logging.error(f"위험 상태 콜백 실행 오류: {e}")

        # 강제 메모리 정리
        self._force_cleanup()

    async def _handle_warning_state(self, usage_mb: float):
        """경고 상태 처리"""
        logging.warning(f"메모리 사용량 경고 수준: {usage_mb:.1f}MB")
        
        # 등록된 콜백 실행
        for callback in self.warning_callbacks:
            try:
                await callback(usage_mb)
            except Exception as e:
                logging.error(f"경고 상태 콜백 실행 오류: {e}")

    async def _handle_cleanup_state(self, usage_mb: float):
        """정리 필요 상태 처리"""
        logging.info(f"메모리 정리 시작: {usage_mb:.1f}MB")
        self._cleanup_old_data()

    def _force_cleanup(self):
        """강제 메모리 정리"""
        import gc
        gc.collect()
        
        # 버퍼 크기 강제 감소
        self._reduce_buffer_sizes()
        
        logging.info("강제 메모리 정리 완료")

    def _cleanup_old_data(self):
        """오래된 데이터 정리"""
        # 사용량 히스토리 정리
        while len(self.usage_history) > 100:  # 최근 100개만 유지
            self.usage_history.popleft()
        
        logging.info("오래된 데이터 정리 완료")

    def _reduce_buffer_sizes(self):
        """버퍼 크기 감소"""
        reduction = self.config.buffer_reduce_percent / 100
        
        # 버퍼 크기 조정 로직 구현
        # (이 부분은 실제 버퍼들과 연동하여 구현 필요)
        
        logging.info(f"버퍼 크기 {self.config.buffer_reduce_percent}% 감소 완료")

    def get_memory_stats(self) -> Dict:
        """메모리 사용 통계"""
        if not self.usage_history:
            return {}
            
        recent_usage = [info['rss'] for info in self.usage_history]
        return {
            'current_usage_mb': recent_usage[-1],
            'average_usage_mb': sum(recent_usage) / len(recent_usage),
            'peak_usage_mb': max(recent_usage),
            'min_usage_mb': min(recent_usage)
        }

    def register_warning_callback(self, callback):
        """경고 상태 콜백 등록"""
        self.warning_callbacks.append(callback)

    def register_critical_callback(self, callback):
        """위험 상태 콜백 등록"""
        self.critical_callbacks.append(callback)