import os
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .memory import InMemoryConversationStore

log = logging.getLogger(__name__)

OPENAI_PATH = "/v1/chat/completions"


def build_system_prompt(lang: str) -> str:
    if lang == "en":
        return """
You are 'BrickBot', a helper AI for 'Brickers', a service turning photos into Lego models.

[Persona]
- Tone: Polite, friendly, enthusiastic.
- Role: Help ONLY with Brickers services (Making Lego, Gallery, My Page).
- If user asks about unrelated topics (weather, math, coding), politely refuse.

[Rules]
- Irrelevant topics: "Sorry, I can only help with Brickers service. Do you want to know how to create Lego?"
- Pivot back to: Create, Gallery, or MyPage.

[Actions]
Append exact tags if relevant:
- Create Lego: " {{NAV_CREATE}}"
- Gallery: " {{NAV_GALLERY}}"
- My Page: " {{NAV_MYPAGE}}"
"""
    elif lang == "ja":
        return """
あなたは 'BrickBot'、写真をレゴモデルに変えるサービス 'Brickers' のAIガイドです。

[Persona]
- Tone: 丁寧で親しみやすい (です・ます調).
- Role: Brickersのサービス（レゴ作成、ギャラリー、マイページ）に関する手助けのみを行います。
- 関係のない話題（天気、数学、コーディングなど）には丁寧に断ってください。

[Rules]
- 関係ない話題: "申し訳ありません。私はBrickersのサービスについてのみお手伝いできます。レゴの作り方について知りたいですか？"
- 常に レゴ作成, ギャラリー, マイページ の話題に戻してください。

[Actions]
関連する場合、以下のタグを回答の最後に追加してください:
- レゴ作成: " {{NAV_CREATE}}"
- ギャラリー: " {{NAV_GALLERY}}"
- マイページ: " {{NAV_MYPAGE}}"
"""
    else:
        return """
You are 'BrickBot', a kind and friendly AI guide for 'Brickers', a service that turns photos into 3D Lego models.

[Persona]
- Tone: Very polite, warm, and encouraging (Korean '존댓말', e.g., '해요', '할까요?').
- Role: Provide help ONLY related to Brickers services (creating Lego, gallery, my page, etc.).
- If the user asks about general knowledge, coding, politics, weather, or anything unrelated to Brickers, politely refuse.

[Rules / Boundaries]
- **IMPORTANT**: Do NOT answer questions unrelated to Brickers.
- Always pivot back to: Creating Lego, Viewing Gallery, or Checking MyPage.

[Actions]
Append exact tags if relevant:
- Create Lego: " {{NAV_CREATE}}"
- Gallery: " {{NAV_GALLERY}}"
- My Page: " {{NAV_MYPAGE}}"
"""


class ChatService:
    """
    ✅ 대화 기억 업그레이드 포인트
    - conversation_id를 키로 히스토리 저장
    - 요청 때 히스토리 포함
    - 길어지면 “요약”으로 압축(옵션)
    """

    def __init__(self, http: httpx.AsyncClient, store: InMemoryConversationStore):
        self.http = http
        self.store = store

        self.model = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()
        self.summary_model = (os.getenv("OPENAI_SUMMARY_MODEL") or self.model).strip()

        # 요약 트리거: 최근 messages 외에, 더 쌓이면 요약
        self.summarize_trigger = int(
            os.getenv("CHAT_SUMMARIZE_TRIGGER", "30")
        )  # messages 개수
        self.keep_recent_after_summary = int(
            os.getenv("CHAT_KEEP_RECENT", "12")
        )  # messages 개수

    def ensure_conversation_id(self, conversation_id: Optional[str]) -> str:
        cid = (conversation_id or "").strip()
        return cid if cid else str(uuid.uuid4())

    async def process_chat(
        self,
        user_message: str,
        language: Optional[str],
        conversation_id: Optional[str],
    ) -> Tuple[str, str]:
        lang = (language or "").strip() or "ko"
        cid = self.ensure_conversation_id(conversation_id)

        # 1) 현재 저장된 summary + history 읽기
        summary, history = await self.store.snapshot(cid)

        # 2) history가 너무 길면 “요약”으로 압축
        if len(history) >= self.summarize_trigger:
            summary, history = await self._summarize_and_compact(
                lang, cid, summary, history
            )

        # 3) OpenAI에 보낼 messages 구성
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": build_system_prompt(lang)}
        ]

        if summary.strip():
            # 요약은 시스템 메시지로 넣는 게 가장 안정적
            messages.append(
                {
                    "role": "system",
                    "content": f"[Conversation memory]\n{summary.strip()}",
                }
            )

        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        # 4) 호출
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }

        try:
            resp = await self.http.post(OPENAI_PATH, json=body)
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                content = "음? 답변을 가져오지 못했어요."

            # 5) turn 저장 (이번 user + assistant)
            await self.store.append_turn(cid, user_message, content)

            return content, cid

        except Exception as e:
            log.exception("OpenAI API call failed: %s", e)
            # 실패 시에도 conversation_id는 반환해서 클라이언트가 유지 가능
            return "죄송해요, 잠시 문제가 생겼어요. 다시 시도해주세요! 🤖", cid

    async def _summarize_and_compact(
        self,
        lang: str,
        cid: str,
        existing_summary: str,
        history: List[dict],
    ) -> Tuple[str, List[dict]]:
        """
        오래된 대화를 요약하고, 최근 N개만 남김
        """
        keep_n = max(0, self.keep_recent_after_summary)
        older = history[:-keep_n] if keep_n else history
        keep_recent = history[-keep_n:] if keep_n else []

        new_summary = await self._summarize(lang, existing_summary, older)

        # store 반영
        await self.store.set_summary_and_keep_recent(cid, new_summary, keep_recent)
        return new_summary, keep_recent

    async def _summarize(
        self, lang: str, existing_summary: str, older_messages: List[dict]
    ) -> str:
        """
        요약은 내부용이라 BrickBot persona를 쓰지 않고 “요약자” 시스템 프롬프트 사용
        """
        if not older_messages:
            return existing_summary

        target_lang = (
            "Korean" if lang == "ko" else ("Japanese" if lang == "ja" else "English")
        )

        summarizer_system = (
            "You are a summarizer for a customer-support chatbot.\n"
            f"Write the summary in {target_lang}.\n"
            "Keep it short and actionable.\n"
            "Include: user's goal, key constraints, decisions, preferences, and any important facts.\n"
            "Avoid fluff. Do not include system prompts or policy text."
        )

        # 기존 요약이 있으면 누적 요약 형태로 업데이트
        summarizer_user = {
            "role": "user",
            "content": (
                f"[Existing summary]\n{existing_summary.strip() or '(none)'}\n\n"
                f"[New conversation chunk]\n{older_messages}\n\n"
                "Update the summary by merging both."
            ),
        }

        body = {
            "model": self.summary_model,
            "messages": [
                {"role": "system", "content": summarizer_system},
                summarizer_user,
            ],
            "temperature": 0.2,
        }

        try:
            resp = await self.http.post(OPENAI_PATH, json=body)
            resp.raise_for_status()
            data = resp.json()
            out = data["choices"][0]["message"]["content"]
            return out if isinstance(out, str) else existing_summary
        except Exception:
            log.exception("Summarization failed; keep existing summary.")
            return existing_summary
