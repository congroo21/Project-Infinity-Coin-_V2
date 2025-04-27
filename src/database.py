# File: src/database.py

# 기본 Python 라이브러리
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

# 데이터베이스 관련
import sqlite3  # 동기식 SQLite 작업용
import aiosqlite  # 비동기식 SQLite 작업용
import asyncio  # 비동기 처리용

# 수치 계산
import numpy as np  # 통계 계산용

# 커스텀 예외
from src.exceptions import DatabaseError

@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    db_path: str = "trading_history.db"
    batch_size: int = 100
    cleanup_days: int = 30
    max_connections: int = 5

class ImprovedDatabaseManager:
    """개선된 데이터베이스 매니저"""
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.batch_queue = []
        self._init_database()
        
    def _init_database(self):
        """데이터베이스 초기화"""
        try:
            # 데이터베이스 디렉토리 생성
            Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.config.db_path) as db:
                # 기존 테이블 삭제
                db.execute("DROP TABLE IF EXISTS trades")
                db.execute("DROP TABLE IF EXISTS performance_metrics")
                db.execute("DROP TABLE IF EXISTS portfolio_states")
                db.execute("DROP TABLE IF EXISTS risk_events")
                db.execute("DROP TABLE IF EXISTS backtest_results")
                
                # 거래 기록 테이블 생성
                db.execute("""
                    CREATE TABLE trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        trade_type TEXT,
                        symbol TEXT,
                        price REAL,
                        volume REAL,
                        fee REAL,
                        slippage REAL,
                        pnl REAL,
                        strategy_id TEXT,
                        timeframe TEXT,
                        position_size REAL,
                        execution_time REAL,
                        market_impact REAL,
                        order_status TEXT
                    )
                """)
                
                # trades 테이블 인덱스 생성
                db.execute("CREATE INDEX idx_trades_timestamp ON trades (timestamp)")
                db.execute("CREATE INDEX idx_trades_symbol ON trades (symbol)")
                
                # 성능 메트릭스 테이블 생성
                db.execute("""
                    CREATE TABLE performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        trade_interval REAL,
                        batch_size INTEGER,
                        confidence_threshold REAL,
                        profit_rate REAL,
                        latency REAL,
                        success_rate REAL,
                        var_95 REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL
                    )
                """)
                db.execute("CREATE INDEX idx_metrics_timestamp ON performance_metrics (timestamp)")
                
                # 포트폴리오 상태 테이블 생성
                db.execute("""
                    CREATE TABLE portfolio_states (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        total_value REAL,
                        cash_balance REAL,
                        position_value REAL,
                        position_count INTEGER,
                        realized_pnl REAL,
                        unrealized_pnl REAL
                    )
                """)
                db.execute("CREATE INDEX idx_portfolio_timestamp ON portfolio_states (timestamp)")
                
                # 리스크 이벤트 테이블 생성
                db.execute("""
                    CREATE TABLE risk_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        event_type TEXT,
                        severity TEXT,
                        description TEXT,
                        action_taken TEXT,
                        resolved BOOLEAN,
                        resolution_time DATETIME
                    )
                """)
                db.execute("CREATE INDEX idx_risk_timestamp ON risk_events (timestamp)")
                
                # 백테스팅 결과 테이블 생성
                db.execute("""
                    CREATE TABLE backtest_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        strategy_id TEXT,
                        start_date DATETIME,
                        end_date DATETIME,
                        total_trades INTEGER,
                        win_rate REAL,
                        profit_factor REAL,
                        sharpe_ratio REAL,
                        max_drawdown REAL,
                        parameters TEXT
                    )
                """)
                db.execute("CREATE INDEX idx_backtest_strategy ON backtest_results (strategy_id)")
                
                db.commit()
                logging.info("Database initialization completed successfully")

        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error during database initialization: {e}")
            raise DatabaseError(f"Unexpected error: {str(e)}")

    async def save_trade(self, trade_data: Dict) -> int:
        """거래 기록 저장"""
        try:
            async with aiosqlite.connect(self.config.db_path) as db:
                cursor = await db.execute("""
                    INSERT INTO trades (
                        trade_type, symbol, price, volume, fee,
                        slippage, pnl, strategy_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data['trade_type'],
                    trade_data['symbol'],
                    trade_data['price'],
                    trade_data['volume'],
                    trade_data.get('fee', 0),
                    trade_data.get('slippage', 0),
                    trade_data.get('pnl', 0),
                    trade_data.get('strategy_id', 'default')
                ))
                await db.commit()
                return cursor.lastrowid
                
        except aiosqlite.Error as e:
            logging.error(f"Database error in save_trade: {e}")
            raise DatabaseError(f"Failed to save trade: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in save_trade: {e}")
            raise DatabaseError(f"Unexpected error: {str(e)}")

    async def save_performance_metrics(self, metrics: Dict):
        """성능 지표 저장 (배치 처리)"""
        try:
            self.batch_queue.append(('performance_metrics', metrics))
            
            if len(self.batch_queue) >= self.config.batch_size:
                await self._process_batch()
                
        except Exception as e:
            logging.error(f"Error in save_performance_metrics: {e}")
            raise DatabaseError(f"Failed to save performance metrics: {str(e)}")

    async def save_portfolio_state(self, state: Dict):
        """포트폴리오 상태 저장"""
        try:
            async with aiosqlite.connect(self.config.db_path) as db:
                await db.execute("""
                    INSERT INTO portfolio_states (
                        total_value, cash_balance, position_value,
                        position_count, realized_pnl, unrealized_pnl
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    state['total_value'],
                    state['cash_balance'],
                    state['position_value'],
                    state['position_count'],
                    state['realized_pnl'],
                    state['unrealized_pnl']
                ))
                await db.commit()
                
        except aiosqlite.Error as e:
            logging.error(f"Database error in save_portfolio_state: {e}")
            raise DatabaseError(f"Failed to save portfolio state: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in save_portfolio_state: {e}")
            raise DatabaseError(f"Unexpected error: {str(e)}")

    async def save_risk_event(self, event: Dict):
        """리스크 이벤트 기록"""
        try:
            async with aiosqlite.connect(self.config.db_path) as db:
                await db.execute("""
                    INSERT INTO risk_events (
                        event_type, severity, description,
                        action_taken, resolved
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    event['event_type'],
                    event['severity'],
                    event['description'],
                    event['action_taken'],
                    False
                ))
                await db.commit()
                
        except aiosqlite.Error as e:
            logging.error(f"Database error in save_risk_event: {e}")
            raise DatabaseError(f"Failed to save risk event: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in save_risk_event: {e}")
            raise DatabaseError(f"Unexpected error: {str(e)}")
    
    def _serialize_for_json(self, obj):
        """JSON 직렬화를 위해 객체를 변환하는 재귀 함수"""
        if isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._serialize_for_json(item) for item in obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return float(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return obj
        
    async def save_risk_state(self, state: Dict):
        """리스크 상태 저장"""
        try:
            async with aiosqlite.connect(self.config.db_path) as db:
                # 전체 상태를 단일 직렬화로 처리
                serialized_state = self._serialize_for_json(state)
                state_json = json.dumps(serialized_state)
                
                # 필요한 필드 추출 및 기본값 설정
                warning_level = serialized_state.get('warning_level', 'normal')
                active_alerts = serialized_state.get('active_alerts', [])
                
                # 요약 정보 생성
                summary = f"경고 수준: {warning_level}, 활성 알림: {len(active_alerts)}개"
                
                # 위험 이벤트로 저장
                await db.execute("""
                    INSERT INTO risk_events (
                        event_type, severity, description, action_taken, resolved
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    'state_update',
                    warning_level,
                    summary,
                    state_json,  # 전체 상태를 action_taken 필드에 저장
                    False
                ))
                await db.commit()
                    
        except aiosqlite.Error as e:
            logging.error(f"리스크 상태 DB 저장 오류: {e}")
        except json.JSONDecodeError as e:
            logging.error(f"리스크 상태 JSON 직렬화 오류: {e}")
        except Exception as e:
            logging.error(f"리스크 상태 저장 예상치 못한 오류: {e}")

    async def update_risk_event(self, event_id: int, resolution_data: Dict):
        """리스크 이벤트 업데이트"""
        try:
            async with aiosqlite.connect(self.config.db_path) as db:
                await db.execute("""
                    UPDATE risk_events
                    SET resolved = ?, resolution_time = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (True, event_id))
                await db.commit()
                
        except aiosqlite.Error as e:
            logging.error(f"Database error in update_risk_event: {e}")
            raise DatabaseError(f"Failed to update risk event: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in update_risk_event: {e}")
            raise DatabaseError(f"Unexpected error: {str(e)}")

    async def save_backtest_result(self, result: Dict):
        """백테스팅 결과 저장"""
        try:
            async with aiosqlite.connect(self.config.db_path) as db:
                # 파라미터 직렬화에도 개선된 함수 사용
                serialized_params = self._serialize_for_json(result['parameters'])
                params_json = json.dumps(serialized_params)
                
                await db.execute("""
                    INSERT INTO backtest_results (
                        strategy_id, start_date, end_date,
                        total_trades, win_rate, profit_factor,
                        sharpe_ratio, max_drawdown, parameters
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result['strategy_id'],
                    result['start_date'],
                    result['end_date'],
                    result['total_trades'],
                    result['win_rate'],
                    result['profit_factor'],
                    result['sharpe_ratio'],
                    result['max_drawdown'],
                    params_json
                ))
                await db.commit()
                
        except aiosqlite.Error as e:
            logging.error(f"Database error in save_backtest_result: {e}")
            raise DatabaseError(f"Failed to save backtest result: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in save_backtest_result: {e}")
            raise DatabaseError(f"Unexpected error: {str(e)}")

    async def _process_batch(self):
        """배치 처리"""
        if not self.batch_queue:
            return
            
        try:
            async with aiosqlite.connect(self.config.db_path) as db:
                await db.execute("BEGIN TRANSACTION")
                try:
                    for table, data in self.batch_queue:
                        if table == 'performance_metrics':
                            await db.execute("""
                                INSERT INTO performance_metrics (
                                    trade_interval, batch_size, confidence_threshold,
                                    profit_rate, latency, success_rate,
                                    var_95, max_drawdown, sharpe_ratio
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                data['trade_interval'],
                                data['batch_size'],
                                data['confidence_threshold'],
                                data['profit_rate'],
                                data['latency'],
                                data['success_rate'],
                                data.get('var_95', 0),
                                data.get('max_drawdown', 0),
                                data.get('sharpe_ratio', 0)
                            ))
                    
                    await db.commit()
                    self.batch_queue.clear()
                    
                except Exception as e:
                    await db.execute("ROLLBACK")
                    raise e
                    
        except aiosqlite.Error as e:
            logging.error(f"Database error in _process_batch: {e}")
            raise DatabaseError(f"Failed to process batch: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in _process_batch: {e}")
            raise DatabaseError(f"Unexpected error: {str(e)}")

    async def cleanup_old_data(self):
        """오래된 데이터 정리"""
        try:
            async with aiosqlite.connect(self.config.db_path) as db:
                cleanup_date = datetime.now() - timedelta(days=self.config.cleanup_days)
                
                for table in ['trades', 'performance_metrics', 'portfolio_states']:
                    await db.execute(f"""
                        DELETE FROM {table}
                        WHERE timestamp < ?
                    """, (cleanup_date,))
                    
                await db.commit()
                logging.info(f"Cleaned up old data before {cleanup_date}")
                
        except aiosqlite.Error as e:
            logging.error(f"Database error in cleanup_old_data: {e}")
            raise DatabaseError(f"Failed to cleanup old data: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in cleanup_old_data: {e}")
            raise DatabaseError(f"Unexpected error: {str(e)}")

    async def get_strategy_performance(
        self,
        strategy_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """전략별 성과 분석"""
        try:
            async with aiosqlite.connect(self.config.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
                        AVG(pnl) as avg_pnl,
                        MAX(price) as max_price,
                        MIN(price) as min_price,
                        SUM(volume) as total_volume,
                        SUM(fee) as total_fees
                    FROM trades
                    WHERE strategy_id = ?
                    AND timestamp BETWEEN ? AND ?
                """, (strategy_id, start_date, end_date))
                
                result = dict(await cursor.fetchone())
                
                # 추가 성능 지표 계산
                cursor = await db.execute("""
                    SELECT pnl
                    FROM trades
                    WHERE strategy_id = ?
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY pnl
                """, (strategy_id, start_date, end_date))
                
                pnl_values = [row[0] for row in await cursor.fetchall()]
                
                if pnl_values:
                    result['var_95'] = float(np.percentile(pnl_values, 5))
                    result['volatility'] = float(np.std(pnl_values))
                    
                    # 샤프 비율 계산 (연간화)
                    risk_free_rate = 0.02  # 2% 연간 무위험 수익률 가정
                    excess_returns = [pnl - risk_free_rate/365 for pnl in pnl_values]
                    if len(excess_returns) > 1:
                        avg_excess_return = np.mean(excess_returns)
                        volatility = np.std(excess_returns)
                        if volatility > 0:
                            result['sharpe_ratio'] = (avg_excess_return / volatility) * np.sqrt(365)
                
                return result
                
        except aiosqlite.Error as e:
            logging.error(f"Database error in get_strategy_performance: {e}")
            raise DatabaseError(f"Failed to get strategy performance: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in get_strategy_performance: {e}")
            raise DatabaseError(f"Unexpected error: {str(e)}")

    async def get_risk_summary(self) -> Dict:
        """리스크 요약 정보"""
        try:
            async with aiosqlite.connect(self.config.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # 활성 리스크 이벤트 조회
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as active_events,
                        MAX(severity) as max_severity
                    FROM risk_events
                    WHERE resolved = 0
                """)
                risk_summary = dict(await cursor.fetchone())
                
                # 포트폴리오 상태 조회
                cursor = await db.execute("""
                    SELECT *
                    FROM portfolio_states
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                portfolio = dict(await cursor.fetchone())
                
                return {
                    'risk_events': risk_summary,
                    'portfolio_state': portfolio
                }
                
        except aiosqlite.Error as e:
            logging.error(f"Database error in get_risk_summary: {e}")
            raise DatabaseError(f"Failed to get risk summary: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in get_risk_summary: {e}")
            raise DatabaseError(f"Unexpected error: {str(e)}")

    async def save_market_data(self, market_data: Dict) -> int:
        """시장 데이터 저장 (이전 버전 호환성 유지)"""
        try:
            async with aiosqlite.connect(self.config.db_path) as db:
                # 간단한 직렬화를 통한 시장 데이터 저장
                timestamp = market_data.get('timestamp', datetime.now())
                price = market_data.get('price', 0.0)
                volume = market_data.get('volume', 0.0)
                
                cursor = await db.execute("""
                    INSERT INTO trades (
                        timestamp, trade_type, price, volume
                    ) VALUES (?, ?, ?, ?)
                """, (
                    timestamp,
                    'market_data',
                    price,
                    volume
                ))
                await db.commit()
                return cursor.lastrowid
                    
        except aiosqlite.Error as e:
            logging.error(f"Database error in save_market_data: {e}")
            raise DatabaseError(f"Failed to save market data: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error in save_market_data: {e}")
            raise DatabaseError(f"Unexpected error: {str(e)}")

# 이전 버전 호환성을 위한 별칭
DatabaseManager = ImprovedDatabaseManager