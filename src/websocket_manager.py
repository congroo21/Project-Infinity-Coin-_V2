import asyncio
import json
import logging
import websockets
import socket
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass
import numpy as np
from multiprocessing import Process, Queue
from queue import Empty  # 명시적으로 Empty import

@dataclass
class WebSocketConfig:
    """웹소켓 설정"""
    uri: str
    ping_interval: float = 0.02        # 20ms로 더욱 단축
    reconnect_delay: float = 0.1       # 재연결 딜레이 최소화
    max_reconnect_attempts: int = 5
    buffer_size: int = 10000           # 버퍼 크기 최적화
    connection_timeout: float = 1.0     # 타임아웃 최적화
    message_timeout: float = 0.05      # 50ms 메시지 타임아웃
    max_message_size: int = 1024 * 1024  # 1MB
    processor_count: int = 4           # 병렬 처리 프로세스 수

@dataclass
class MarketTickData:
    """시장 데이터 틱"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    type: str  # 'trade', 'orderbook', 'ticker'
    raw_data: Dict
    sequence_id: int
    
    def validate(self) -> bool:
        """데이터 유효성 검증"""
        try:
            if not isinstance(self.timestamp, datetime):
                return False
            if not self.symbol:
                return False
            if not isinstance(self.price, (int, float)) or self.price <= 0:
                return False
            if not isinstance(self.volume, (int, float)) or self.volume < 0:
                return False
            return True
        except Exception:
            return False

class DataProcessor(Process):
    """데이터 처리 프로세스"""
    def __init__(self, input_queue: Queue, output_queue: Queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.processed_count = 0
        self.error_count = 0

    def run(self):
        """프로세스 실행"""
        while self.running:
            try:
                # 짧은 타임아웃으로 메시지 확인
                try:
                    message = self.input_queue.get(timeout=0.1)
                    if message is None:  # 종료 시그널
                        self.running = False
                        break
                        
                    processed_data = self._process_message(message)
                    if processed_data:
                        self.output_queue.put(processed_data)
                        self.processed_count += 1
                except Empty:
                    # 입력 큐가 일정 시간 동안 비어있으면 다시 확인
                    continue
                    
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                self.error_count += 1
                print(f"데이터 처리 오류: {e}")

    # DataProcessor 클래스의 _process_message 메서드를 더 엄격하게 수정
    def _process_message(self, message: str) -> Optional[Dict]:
        """메시지 처리 및 엄격한 검증"""
        try:
            if isinstance(message, str):
                data = json.loads(message)
            else:
                data = message
                
            # 필수 필드 검증 추가
            required_fields = ['type', 'data', 'timestamp']
            if not all(field in data for field in required_fields):
                logging.warning(f"필수 필드 누락: {data}")
                return None
                
            # 타입 검증 추가
            if data['type'] not in ['trade', 'orderbook', 'ticker']:
                logging.warning(f"알 수 없는 메시지 타입: {data['type']}")
                return None

            data['processed_at'] = datetime.now().isoformat()
            return data
                
        except json.JSONDecodeError:
            logging.error("JSON 디코딩 실패")
            return None
        except Exception as e:
            logging.error(f"메시지 처리 실패: {e}")
            return None

    def stop(self):
        """프로세스 안전 종료"""
        self.running = False
        try:
            # 종료 시그널 전송
            self.input_queue.put(None, timeout=0.1)
        except Exception:
            pass

    def get_stats(self) -> dict:
        """
        프로세서의 간단한 상태/성능 통계를 반환.
        외부(Manager)에서 각 프로세서별 처리량을 확인할 때 사용한다.
        """
        return {
            "processed_count": self.processed_count,
            "error_count": self.error_count
        }

class WebSocketManager:
    """고성능 웹소켓 관리자"""
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.websocket = None
        self.connected = False
        self.running = False
        self.connection_attempts = 0
        
        # 버퍼 및 카운터
        self.message_buffer = deque(maxlen=config.buffer_size)
        self.trade_buffer = deque(maxlen=config.buffer_size)
        self.orderbook_buffer = deque(maxlen=config.buffer_size)
        self.sequence_counter = 0
        
        # 성능 모니터링
        self.latency_records = deque(maxlen=1000)
        self.message_count = 0
        self.error_count = 0
        self.start_time = datetime.now()
        self.last_ping_time = None
        
        # 메시지 처리 큐
        self.message_queue = asyncio.Queue()
        self.processing_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        
        # 병렬 처리 설정
        self.input_queues = [Queue() for _ in range(config.processor_count)]
        self.output_queue = Queue()
        self.processors = [
            DataProcessor(input_queue, self.output_queue)
            for input_queue in self.input_queues
        ]
        
        self._setup_logging()

    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    async def connect(self) -> bool:
        """최적화된 웹소켓 연결"""
        if self.connected:
            return True

        try:
            # 웹소켓 연결 설정
            self.websocket = await websockets.connect(
                self.config.uri,
                ping_interval=self.config.ping_interval,
                max_size=self.config.max_message_size,
                close_timeout=self.config.connection_timeout
            )
            
            # TCP 최적화
            transport = self.websocket.transport
            if transport:
                sock = transport.get_extra_info('socket')
                if sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
            self.connected = True
            self.last_ping_time = datetime.now()
            logging.info("WebSocket 연결 성공")
            return True
            
        except Exception as e:
            logging.error(f"WebSocket 연결 실패: {e}")
            return False
        
    async def send_message(self, message: Any):
        """웹소켓 메시지 전송"""
        if not self.connected:
            logging.error("웹소켓이 연결되지 않음")
            return
            
        try:
            await self.websocket.send(json.dumps(message))
            logging.info("구독 요청 전송 완료")
        except Exception as e:
            logging.error(f"메시지 전송 실패: {e}")    

    async def start(self):
        """데이터 수신 시작"""
        if not await self.connect():
            raise ConnectionError("WebSocket 연결 실패")

        self.running = True
        
        # 프로세서 시작
        for processor in self.processors:
            processor.start()
        
        # 작업자 태스크 생성
        self.tasks = [
            asyncio.create_task(self._message_receiver()),
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._result_handler()),
            asyncio.create_task(self._performance_monitor())
        ]
        
        logging.info("WebSocket 매니저 시작됨")

    async def _try_reconnect(self) -> bool:
        """재연결 시도 로직 개선"""
        retries_left = self.config.max_reconnect_attempts
        base_delay = self.config.reconnect_delay
        
        while retries_left > 0 and not self.connected:
            try:
                attempt = self.config.max_reconnect_attempts - retries_left + 1
                wait_time = base_delay * min(2 ** (attempt - 1), 10)  # 지수 백오프 (최대 10배)
                
                logging.info(f"재연결 시도 {attempt}/{self.config.max_reconnect_attempts} "
                             f"({wait_time:.2f}초 대기 중)")
                
                # 재연결 전 대기
                await asyncio.sleep(wait_time)
                
                # 소켓 상태 확인 및 정리
                if self.websocket and self.websocket.open:
                    await self.websocket.close()
                
                # 연결 시도
                if await self.connect():
                    logging.info(f"재연결 성공 (시도 {attempt}/{self.config.max_reconnect_attempts})")
                    self.connection_attempts = 0
                    return True
                    
                retries_left -= 1
                
            except Exception as e:
                logging.error(f"재연결 시도 {attempt} 중 오류 발생: {e}")
                retries_left -= 1
                # 짧은 대기 후 다음 시도
                await asyncio.sleep(1)
        
        if not self.connected:
            logging.critical(f"최대 재연결 시도 횟수({self.config.max_reconnect_attempts})를 초과했습니다. 재연결 실패.")
        
        return self.connected

    async def _message_receiver(self):
        """메시지 수신 최적화"""
        current_processor = 0  # 현재 프로세서 인덱스 초기화
        max_consecutive_errors = 5
        consecutive_errors = 0
        
        while self.running:
            try:
                if not self.connected:
                    reconnect_success = await self._try_reconnect()
                    if not reconnect_success:
                        # 재연결 실패 시 일정 시간 대기 후 다시 시도
                        await asyncio.sleep(5)
                        continue

                # 메시지 수신 시도 로깅 추가
                logging.debug("메시지 수신 대기 중...")
                
                # 메시지 수신 (타임아웃 설정)
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=self.config.message_timeout
                    )
                    # 메시지 수신 성공 시 연속 오류 카운터 초기화
                    consecutive_errors = 0
                    
                    # 메시지 수신 성공 로깅 추가
                    logging.info(f"메시지 수신: {message[:200]}...")  # 처음 200자만 로깅
                    receive_time = datetime.now()
                    
                    # 레이턴시 측정
                    if self.last_ping_time:
                        latency = (receive_time - self.last_ping_time).total_seconds() * 1000
                        self.latency_records.append(latency)

                    # 메시지 전처리 및 분배
                    processed_message = self._preprocess_message(message)
                    if processed_message:
                        self.input_queues[current_processor].put(processed_message)
                        current_processor = (current_processor + 1) % len(self.processors)
                        self.message_count += 1
                        
                except asyncio.TimeoutError:
                    # 타임아웃 발생 시 연결 상태 확인
                    if self.websocket:
                        try:
                            # 핑 메시지 전송으로 연결 확인
                            pong_waiter = await self.websocket.ping()
                            await asyncio.wait_for(pong_waiter, timeout=5)
                            # 핑 성공 시 계속 진행
                            continue
                        except:
                            # 핑 실패 시 연결 끊김으로 간주
                            self.connected = False
                            logging.warning("핑 실패로 연결 끊김 감지")
                        continue

            except websockets.exceptions.ConnectionClosed:
                logging.warning("WebSocket 연결 끊김")
                self.connected = False
                consecutive_errors += 1
                if consecutive_errors > max_consecutive_errors:
                    logging.error(f"연속 {max_consecutive_errors}회 연결 오류 발생")
                    # 일정 시간 대기 후 재시도
                    await asyncio.sleep(10)
                    consecutive_errors = 0
                    
            except Exception as e:
                self.error_count += 1
                consecutive_errors += 1
                logging.error(f"메시지 수신 오류: {e}")
                if consecutive_errors > max_consecutive_errors:
                    logging.error(f"연속 {max_consecutive_errors}회 오류 발생, 대기 후 재시도")
                    await asyncio.sleep(10)
                    consecutive_errors = 0
                else:
                    await asyncio.sleep(0.1)

    def _preprocess_message(self, message: str) -> Optional[Dict]:
        """메시지 전처리"""
        try:
            data = json.loads(message)
            
            # 기본 메타데이터 추가
            data.update({
                'received_at': datetime.now().isoformat(),
                'sequence_id': self.sequence_counter
            })
            self.sequence_counter += 1
            
            return data
            
        except json.JSONDecodeError:
            return None
        except Exception as e:
            logging.error(f"메시지 전처리 오류: {e}")
            return None

    async def _message_processor(self):
        """메시지 처리 최적화"""
        from queue import Empty  # Empty import 추가
        
        batch_size = 100  # 배치 처리 크기
        batch = []
        
        while self.running:
            try:
                # 출력 큐에서 처리된 메시지 수집
                while len(batch) < batch_size:
                    try:
                        message = self.output_queue.get_nowait()
                        if message:
                            batch.append(message)
                    except Empty:  # Queue.Empty 대신 Empty 사용
                        break

                if batch:
                    # 배치 처리
                    processed_batch = await self._process_message_batch(batch)
                    
                    # 결과 분류 및 저장
                    for processed_message in processed_batch:
                        message_type = processed_message.get('type', 'unknown')
                        if message_type == 'trade':
                            self.trade_buffer.append(processed_message)
                        elif message_type == 'orderbook':
                            self.orderbook_buffer.append(processed_message)
                        
                        # 결과 큐에 추가
                        await self.result_queue.put(processed_message)
                    
                    batch = []  # 배치 초기화
                
                await asyncio.sleep(0.001)  # 1ms 대기
                
            except Exception as e:
                logging.error(f"메시지 처리 오류: {e}")
                await asyncio.sleep(0.1)

    async def _process_message_batch(self, messages: List[Dict]) -> List[Dict]:
        """배치 메시지 처리"""
        processed_messages = []
        
        for message in messages:
            try:
                # 메시지 정규화 및 보강
                processed_message = {
                    'timestamp': message.get('received_at'),
                    'type': message.get('type', 'unknown'),
                    'data': message.get('data', {}),
                    'processed_at': datetime.now().isoformat(),
                    'sequence_id': message.get('sequence_id'),
                    'processing_time_ms': self._calculate_processing_time(
                        message.get('received_at')
                    )
                }
                
                processed_messages.append(processed_message)
                
            except Exception as e:
                logging.error(f"단일 메시지 처리 오류: {e}")
                continue
        
        return processed_messages

    def _calculate_processing_time(self, received_time_str: str) -> float:
        """처리 시간 계산"""
        try:
            received_time = datetime.fromisoformat(received_time_str)
            processing_time = (datetime.now() - received_time).total_seconds() * 1000
            return round(processing_time, 2)
        except Exception:
            return 0.0

    async def _result_handler(self):
        """결과 처리"""
        while self.running:
            try:
                result = await self.result_queue.get()
                if 'processing_time_ms' in result:
                    processing_time = result['processing_time_ms']
                    if processing_time > 100:  # 100ms 이상 걸린 경우
                        logging.warning(
                            f"긴 처리 시간 감지: {processing_time}ms, "
                            f"Type: {result.get('type')}, "
                            f"Sequence: {result.get('sequence_id')}"
                        )
                
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)
            except Exception as e:
                logging.error(f"결과 처리 오류: {e}")
                await asyncio.sleep(0.1)

    async def _performance_monitor(self):
        """성능 모니터링"""
        while self.running:
            try:
                # 기본 메트릭 수집
                current_stats = self._collect_performance_stats()
                
                # 프로세서 상태 확인
                processor_stats = self._collect_processor_stats()
                
                # 버퍼 관리 호출 추가
                self._manage_buffers()  # 여기에 추가
                
                # 통합 리포트 생성
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'basic_metrics': current_stats,
                    'processor_metrics': processor_stats
                }
                
                # 성능 메트릭 로깅
                self._log_performance_metrics(report)
                
                await asyncio.sleep(5)  # 5초마다 모니터링
                
            except Exception as e:
                logging.error(f"성능 모니터링 오류: {e}")
                await asyncio.sleep(1)

    def _collect_performance_stats(self) -> Dict:
        """성능 통계 수집"""
        stats = {
            'message_count': self.message_count,
            'error_count': self.error_count,
            'success_rate': (
                (self.message_count - self.error_count) / 
                max(1, self.message_count)
            ) * 100,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }
        
        # 레이턴시 통계
        if self.latency_records:
            latency_array = np.array(self.latency_records)
            stats.update({
                'latency': {
                    'mean': np.mean(latency_array),
                    'median': np.median(latency_array),
                    'p95': np.percentile(latency_array, 95),
                    'p99': np.percentile(latency_array, 99),
                    'max': np.max(latency_array)
                }
            })
            
        return stats

    def _collect_processor_stats(self) -> List[Dict]:
        """프로세서 상태 수집"""
        return [processor.get_stats() for processor in self.processors]

    # WebSocketManager 클래스의 _log_performance_metrics 메서드 수정
    def _log_performance_metrics(self, report: Dict):
        """향상된 성능 메트릭 로깅"""
        basic_metrics = report['basic_metrics']
        latency = basic_metrics.get('latency', {})
        processor_metrics = report['processor_metrics']
        
        # 처리량 계산
        messages_per_second = (
            basic_metrics['message_count'] / 
            max(1, basic_metrics['uptime_seconds'])
        )
        
        # 프로세서 효율성
        processor_efficiency = sum(
            p['processed_count'] for p in processor_metrics
        ) / max(1, len(processor_metrics))
        
        logging.info(
            f"성능 메트릭 ==="
            f"\n처리량: {messages_per_second:.1f} msgs/sec"
            f"\n에러율: {(basic_metrics['error_count']/max(1, basic_metrics['message_count'])*100):.2f}%"
            f"\n평균 지연: {latency.get('mean', 0):.2f}ms"
            f"\nP95 지연: {latency.get('p95', 0):.2f}ms"
            f"\n프로세서 효율성: {processor_efficiency:.1f} msgs/processor"
            f"\n버퍼 상태: {basic_metrics.get('buffer_sizes', {})}"
        )

    # WebSocketManager 클래스에 버퍼 관리 메서드 추가
    def _manage_buffers(self):
        """버퍼 상태 관리 및 최적화"""
        # 버퍼가 80% 이상 찼을 때 경고
        for name, buffer in [
            ('message', self.message_buffer),
            ('trade', self.trade_buffer),
            ('orderbook', self.orderbook_buffer)
        ]:
            usage = len(buffer) / buffer.maxlen
            if usage > 0.8:
                logging.warning(f"{name} 버퍼 사용량 높음: {usage:.1%}")
                
                # 오래된 데이터 정리 검토
                if name != 'message':  # 메시지 버퍼는 제외
                    while len(buffer) > buffer.maxlen * 0.6:  # 60%까지 정리
                        buffer.popleft()    

    async def stop(self):
        """안전한 종료 처리"""
        self.running = False
        
        # 작업자 태스크 종료
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # 프로세서 종료
        for processor in self.processors:
            processor.stop()
            processor.join()
        
        # 웹소켓 연결 종료
        if self.websocket:
            await self.websocket.close()
            
        self.connected = False
        logging.info("WebSocket 매니저가 안전하게 종료됨")

    def get_status(self) -> Dict:
        """현재 상태 정보"""
        return {
            'connected': self.connected,
            'running': self.running,
            'message_count': self.message_count,
            'error_count': self.error_count,
            'buffer_sizes': {
                'trade': len(self.trade_buffer),
                'orderbook': len(self.orderbook_buffer)
            },
            'uptime': str(datetime.now() - self.start_time)
        }