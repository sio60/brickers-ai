# ============================================================================
# LLM 클라이언트 추상화 모듈
# 다양한 LLM API(Groq, OpenAI, Gemini 등)를 통일된 인터페이스로 사용할 수 있게 함
# 나중에 다른 LLM으로 교체할 때 이 파일만 수정하면 됨
# ============================================================================

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path

# .env 파일에서 환경변수 자동 로드
try:
    from dotenv import load_dotenv
    
    # 프로젝트 루트의 .env 파일 찾기
    _THIS_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _THIS_DIR.parent.parent  # brick-engine/agent -> brick-engine -> brickers-ai
    _ENV_PATH = _PROJECT_ROOT / ".env"
    
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH, override=True) # override=True로 설정하여 강제로 덮어씀
        print(f"[LLM] .env 파일 로드됨: {_ENV_PATH}")
        
        # 랭스미스 연동 확인 로그
        tracing = os.environ.get("LANGCHAIN_TRACING_V2")
        project = os.environ.get("LANGCHAIN_PROJECT")
        api_key = os.environ.get("LANGCHAIN_API_KEY")
        
        if tracing == "true":
            masked_key = f"{api_key[:10]}..." if api_key else "None"
            print(f"[LLM] 랭스미스 트레이싱 활성화됨 (Project: {project}, API Key: {masked_key})")
        else:
            print("[LLM] 랭스미스 트레이싱이 비활성화 상태입니다. (.env 설정을 확인하세요)")
    else:
        print(f"[LLM] .env 파일을 찾을 수 없습니다: {_ENV_PATH}")
except ImportError:
    pass  # python-dotenv가 없으면 OS 환경변수만 사용
except Exception as e:
    print(f"[LLM] .env 로드 중 에러 발생: {e}")


class BaseLLMClient(ABC):
    """
    LLM 클라이언트 기본 인터페이스
    새로운 LLM 제공자를 추가하려면 이 클래스를 상속받아 구현
    """
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        LLM에게 프롬프트를 전달하고 응답을 받음
        
        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (LLM의 역할 설정)
            
        Returns:
            LLM 응답 텍스트
        """
        pass
    
    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """
        LLM에게 프롬프트를 전달하고 JSON 응답을 파싱해서 반환
        
        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트
            
        Returns:
            파싱된 JSON 딕셔너리
        """
        pass


class GroqClient(BaseLLMClient):
    """
    Groq API 클라이언트
    Llama 3.3 70B 모델을 사용하여 빠른 응답 제공
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Groq 클라이언트 초기화
        
        Args:
            api_key: Groq API 키 (없으면 환경변수 GROQ_API_KEY에서 읽음)
            model: 사용할 모델명 (기본값: llama-3.3-70b-versatile)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API 키가 필요합니다. "
                "환경변수 GROQ_API_KEY를 설정하거나 api_key 파라미터를 전달하세요.\n"
                "API 키 발급: https://console.groq.com/keys"
            )
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Groq 클라이언트 lazy 초기화"""
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "groq 패키지가 설치되지 않았습니다. "
                    "다음 명령어로 설치하세요: pip install groq"
                )
        return self._client
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Groq API를 통해 텍스트 생성
        """
        client = self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        
        return response.choices[0].message.content
    
    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """
        Groq API를 통해 JSON 응답 생성
        JSON 형식을 강제하기 위해 프롬프트에 지시 추가
        """
        # JSON 출력을 강제하는 추가 지시
        json_instruction = "\n\n반드시 유효한 JSON 형식으로만 응답하세요. 다른 설명 없이 JSON만 출력하세요."
        
        full_system = system_prompt + json_instruction if system_prompt else json_instruction.strip()
        
        response_text = self.generate(prompt, full_system)
        
        # JSON 파싱 시도
        try:
            # 코드 블록 제거 (```json ... ``` 형태로 올 수 있음)
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                # 첫 줄과 마지막 줄 제거
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])
            
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[경고] JSON 파싱 실패: {e}")
            print(f"원본 응답: {response_text[:500]}...")
            # 파싱 실패 시 빈 딕셔너리 + 오류 정보 반환
            return {"error": str(e), "raw_response": response_text}


# ============================================================================
# 향후 확장을 위한 다른 LLM 클라이언트 템플릿
# ============================================================================

class GeminiClient(BaseLLMClient):
    """
    Google Gemini API 클라이언트
    gemini-1.5-flash 모델을 기본으로 하여 빠르고 경량화된 응답 제공
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-flash-latest"):
        """
        Gemini 클라이언트 초기화
        
        Args:
            api_key: Google API 키 (없으면 GOOGLE_API_KEY 또는 GEMINI_API_KEY에서 읽음)
            model: 사용할 모델명 (기본값: gemini-flash-latest)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API 키가 필요합니다. "
                "환경변수 GOOGLE_API_KEY를 설정하거나 api_key 파라미터를 전달하세요."
            )
        self.model_name = model
        self._model = None

    def _get_model(self):
        """LangChain 기반 Gemini 모델 lazy 초기화"""
        if self._model is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._model = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=0.7,
                )
            except ImportError:
                raise ImportError(
                    "langchain-google-genai 패키지가 설치되지 않았습니다. "
                    "pip install langchain-google-genai 명령어로 설치하세요."
                )
        return self._model

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        LangChain을 통해 텍스트 생성 (자동 Tracing 포함)
        """
        llm = self._get_model()
        
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        response = llm.invoke(messages)
        content = response.content
        
        # content가 리스트인 경우 (멀티파트 등) 문자열로 변합
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    texts.append(part["text"])
                elif isinstance(part, str):
                    texts.append(part)
            content = "".join(texts)
            
        return content

    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """
        LangChain을 통해 JSON 응답 생성 및 파싱
        """
        # JSON 출력을 강제하는 추가 지시
        json_instruction = "\n\n반드시 유효한 JSON 형식으로만 응답하세요. 다른 설명 없이 JSON만 출력하세요."
        
        full_system = system_prompt + json_instruction if system_prompt else json_instruction.strip()
        
        response_text = self.generate(prompt, full_system)
        
        # JSON 파싱 시도
        try:
            cleaned = response_text.strip()
            # 코드 블록 제거 (```json ... ``` 형태로 올 수 있음)
            if "```" in cleaned:
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json")[1].split("```")[0].strip()
                else:
                    cleaned = cleaned.split("```")[1].split("```")[0].strip()
            
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"[경고] Gemini JSON 파싱 실패: {e}")
            return {"error": str(e), "raw_response": response_text}




class OpenAIClient(BaseLLMClient):
    """
    OpenAI API 클라이언트 (미구현 - 필요시 구현)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        raise NotImplementedError("OpenAI 클라이언트는 아직 구현되지 않았습니다. Groq을 사용하세요.")
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        raise NotImplementedError()
    
    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        raise NotImplementedError()
