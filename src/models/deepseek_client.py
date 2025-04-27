import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
from openai import OpenAI

@dataclass
class MarketAnalysisPrompt:
    """고빈도 거래를 위한 시장 분석 프롬프트"""
    current_price: float
    price_changes: List[float]  # 최근 10초 간의 가격 변화율
    volume_profile: List[float]  # 최근 10초 간의 거래량
    orderbook_imbalance: float  # 호가창 불균형 (-1 ~ 1)
    spread: float              # 스프레드 (%)
    timestamp: datetime

    def to_prompt(self) -> str:
        """HFT용 분석 프롬프트 생성"""
        # 가격 변화 트렌드 계산
        price_trend = sum(1 if change > 0 else -1 for change in self.price_changes)
        volume_trend = sum(self.volume_profile) / len(self.volume_profile) if self.volume_profile else 0

        return (
            f"고빈도 거래 분석 (timestamp: {self.timestamp.isoformat()}):\n"
            f"1. 현재 상태:\n"
            f"   - 가격: {self.current_price:,.0f}원\n"
            f"   - 10초 추세: {price_trend:+d}\n"
            f"   - 거래량 추세: {volume_trend:,.2f}\n"
            f"   - 호가 불균형: {self.orderbook_imbalance:+.3f}\n"
            f"   - 스프레드: {self.spread:.3f}%\n\n"
            f"다음 항목들에 대해 숫자로만 응답하세요:\n"
            f"1. 매수강도(0-100): \n"
            f"2. 매도강도(0-100): \n"
            f"3. 단기방향(-1,0,1): \n"
            f"4. 실행시급성(0-100): \n"
            f"5. 위험도(0-100): \n"
        )

class DeepSeekClient:
    """최적화된 DeepSeek API 클라이언트"""
    def __init__(self, config):
        self.config = config
        self.client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_endpoint
        )
        self.retry_config = config.retry_config
        self.response_cache = {}
        self.last_analysis_time = None
        self.min_analysis_interval = 0.1  # 최소 분석 간격 (초)
        logging.info("DeepSeek 클라이언트 초기화 완료")

    async def analyze_market(self, prompt_data: MarketAnalysisPrompt) -> Dict:
        """최적화된 시장 분석"""
        # 분석 간격 제어
        current_time = datetime.now()
        if (self.last_analysis_time and 
            (current_time - self.last_analysis_time).total_seconds() < self.min_analysis_interval):
            return self.response_cache.get('last_analysis', self._get_default_analysis())

        prompt = prompt_data.to_prompt()
        
        for attempt in range(self.retry_config['max_retries']):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,  # 토큰 수 최소화
                    temperature=0.1  # 일관성 높임
                )
                
                analysis_result = self._parse_numeric_response(response.choices[0].message.content)
                
                # 캐시 업데이트
                self.response_cache['last_analysis'] = analysis_result
                self.last_analysis_time = current_time
                
                return analysis_result
                
            except Exception as e:
                if attempt == self.retry_config['max_retries'] - 1:
                    logging.error(f"시장 분석 실패: {e}")
                    return self._get_default_analysis()

                delay = min(
                    self.retry_config['base_delay'] * (2 ** attempt),
                    self.retry_config['max_delay']
                )
                await asyncio.sleep(delay)

    def _parse_numeric_response(self, content: str) -> Dict:
        """숫자 응답 파싱"""
        try:
            lines = content.strip().split('\n')
            values = []
            
            for line in lines:
                # 숫자만 추출
                nums = [float(s) for s in line.split() if s.replace('-', '').replace('.', '').isdigit()]
                if nums:
                    values.append(nums[0])

            if len(values) >= 5:
                return {
                    'buy_strength': values[0],
                    'sell_strength': values[1],
                    'direction': values[2],
                    'urgency': values[3],
                    'risk': values[4],
                    'timestamp': datetime.now()
                }
            
            return self._get_default_analysis()
            
        except Exception as e:
            logging.error(f"응답 파싱 실패: {e}")
            return self._get_default_analysis()

    def _get_default_analysis(self) -> Dict:
        """기본 분석 결과"""
        return {
            'buy_strength': 0,
            'sell_strength': 0,
            'direction': 0,
            'urgency': 0,
            'risk': 50,
            'timestamp': datetime.now()
        }

    async def cleanup(self):
        """리소스 정리"""
        self.response_cache.clear()
        logging.info("DeepSeek 클라이언트 종료")