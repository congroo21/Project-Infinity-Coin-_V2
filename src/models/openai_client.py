from openai import OpenAI, OpenAIError  # ✅ 최신 OpenAI 라이브러리 예외 처리 방식 적용
import openai
import os

class OpenAIClient:
    def __init__(self):
        """ OpenAI API 클라이언트 초기화 """
        self.client = OpenAI()
        self.model = "gpt-4o"  # ✅ 최신 지원 모델 사용

    def generate_response(self, prompt: str):
        """ OpenAI GPT-4o 모델을 사용하여 응답 생성 (동기) """
        try:
            if not prompt.strip():  # ✅ 빈 프롬프트 입력 처리
                return "No content provided"

            if len(prompt.split()) > 8000:  # ✅ 8000 단어 이상이면 실패 처리
                return "OpenAI API 호출 실패"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=30  # ✅ API 호출 타임아웃 설정 (30초)
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            print(f"❌ OpenAI API 호출 오류: {e}")
            return "OpenAI API 호출 실패"
        except Exception as e:
            print(f"❌ 기타 오류 발생: {e}")
            return "오류 발생"

    def get_embedding(self, text: str):
        """ 텍스트 임베딩 생성 """
        try:
            if not text.strip():
                return None

            if len(text.split()) > 8000:
                return None

            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                timeout=10
            )
            return response.data[0].embedding
        except OpenAIError as e:
            print(f"❌ OpenAI 임베딩 API 호출 오류: {e}")
            return None
        except Exception as e:
            print(f"❌ 기타 오류 발생: {e}")
            return None

# ✅ 예제 사용 (테스트 코드)
if __name__ == "__main__":
    openai_client = OpenAIClient()
    
    # 기본 테스트
    response = openai_client.generate_response("Explain high-frequency trading in simple terms.")
    print(f"📢 GPT-4o 응답: {response}")

    # 빈 프롬프트 테스트
    empty_response = openai_client.generate_response("")
    print(f"📢 빈 프롬프트 테스트 응답: {empty_response}")

    # 긴 프롬프트 테스트 (에러 발생 확인)
    long_prompt = "This is a long input. " * 9000  # 9000 단어
    long_response = openai_client.generate_response(long_prompt)
    print(f"📢 긴 프롬프트 테스트 응답: {long_response}")

    # 텍스트 임베딩 테스트
    embedding = openai_client.get_embedding("High-frequency trading strategy")
    print(f"📢 임베딩 벡터 길이: {len(embedding) if embedding else 'None'}")
