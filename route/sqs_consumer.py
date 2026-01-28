"""
SQS Consumer - AI Server에서 Backend로부터 REQUEST 메시지 수신
"""
from __future__ import annotations

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any

from route.kids_render import process_kids_request_internal
from route.sqs_producer import send_result_message


def log(msg: str) -> None:
    """타임스탬프 포함 로그 출력"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")


def _is_truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# 환경 변수
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2").strip()
SQS_REQUEST_QUEUE_URL = os.environ.get("AWS_SQS_REQUEST_QUEUE_URL", "").strip()  # Backend → AI (REQUEST 수신)
SQS_ENABLED = _is_truthy(os.environ.get("AWS_SQS_ENABLED", "false"))
SQS_POLL_INTERVAL = int(os.environ.get("SQS_POLL_INTERVAL", "5"))  # 초
SQS_MAX_MESSAGES = int(os.environ.get("SQS_MAX_MESSAGES", "1"))  # AI는 CPU intensive → 순차 처리
SQS_WAIT_TIME = int(os.environ.get("SQS_WAIT_TIME", "10"))  # Long polling
SQS_VISIBILITY_TIMEOUT = int(os.environ.get("SQS_VISIBILITY_TIMEOUT", "1800"))  # 30분 (AI 처리 최대 시간)

# boto3 lazy import
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore


_SQS_CLIENT = None


def _get_sqs_client():
    """SQS Client 싱글톤"""
    global _SQS_CLIENT
    if _SQS_CLIENT is not None:
        return _SQS_CLIENT

    if not SQS_ENABLED:
        raise RuntimeError("SQS is not enabled (AWS_SQS_ENABLED=false)")

    if boto3 is None:
        raise RuntimeError("boto3 is not installed (pip install boto3)")

    if not SQS_REQUEST_QUEUE_URL:
        raise RuntimeError("AWS_SQS_REQUEST_QUEUE_URL is not set")

    # boto3는 아래 env를 자동으로 읽음:
    # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
    _SQS_CLIENT = boto3.client("sqs", region_name=AWS_REGION)
    return _SQS_CLIENT


_POLL_COUNT = 0  # 폴링 횟수 추적

async def poll_and_process():
    """
    SQS에서 REQUEST 메시지 폴링 및 처리
    - Long polling 사용 (불필요한 요청 최소화)
    - REQUEST 타입 메시지만 처리
    - 순차 처리 (AI는 CPU intensive)
    """
    global _POLL_COUNT

    if not SQS_ENABLED:
        return

    try:
        sqs = _get_sqs_client()
        _POLL_COUNT += 1

        # 10회마다 폴링 로그 (너무 많은 로그 방지)
        if _POLL_COUNT % 10 == 1:
            log(f"🔄 [SQS Consumer] 폴링 중... (poll #{_POLL_COUNT})")

        response = sqs.receive_message(
            QueueUrl=SQS_REQUEST_QUEUE_URL,
            MaxNumberOfMessages=SQS_MAX_MESSAGES,
            WaitTimeSeconds=SQS_WAIT_TIME,
            VisibilityTimeout=SQS_VISIBILITY_TIMEOUT,
        )

        messages = response.get("Messages", [])

        if messages:
            log(f"📥 [SQS Consumer] 메시지 수신! | count={len(messages)} | poll #{_POLL_COUNT}")
            for msg in messages:
                log(f"   - MessageId: {msg.get('MessageId', 'N/A')}")
                try:
                    body_preview = msg.get("Body", "")[:200]
                    log(f"   - Body preview: {body_preview}...")
                except:
                    pass

        for message in messages:
            await process_message(message)

    except Exception as e:
        log(f"❌ [SQS Consumer] 폴링 실패 | poll #{_POLL_COUNT} | error={str(e)}")


async def process_message(message: Dict[str, Any]):
    """
    메시지 처리
    - REQUEST 타입 메시지만 처리
    - kids_render 내부 함수 호출
    - RESULT 메시지 전송 (성공/실패)
    """
    message_id = message.get("MessageId", "unknown")
    receipt_handle = message["ReceiptHandle"]

    log("=" * 60)
    log(f"📨 [SQS Consumer] 메시지 처리 시작 | messageId={message_id}")

    try:
        body = json.loads(message["Body"])
        log(f"   - type: {body.get('type')}")
        log(f"   - jobId: {body.get('jobId')}")
        log(f"   - sourceImageUrl: {body.get('sourceImageUrl', '')[:80]}...")
        log(f"   - age: {body.get('age')}, budget: {body.get('budget')}")

        # REQUEST 타입만 처리
        if body.get("type") != "REQUEST":
            log(f"⚠️ [SQS Consumer] RESULT 타입 메시지 무시 | messageId={message_id} | type={body.get('type')}")
            delete_message(receipt_handle)
            return

        job_id = body.get("jobId")
        source_image_url = body.get("sourceImageUrl")
        age = body.get("age", "6-7")
        budget = body.get("budget")

        log(f"📌 [SQS Consumer] REQUEST 메시지 처리 시작 | jobId={job_id}")
        log(f"🚀 [SQS Consumer] AI 렌더링 시작...")

        # ✅ Kids 렌더링 실행
        result = await process_kids_request_internal(
            job_id=job_id,
            source_image_url=source_image_url,
            age=age,
            budget=budget,
        )

        log(f"✅ [SQS Consumer] AI 렌더링 완료!")
        log(f"   - correctedUrl: {result.get('correctedUrl', '')[:60]}...")
        log(f"   - modelUrl: {result.get('modelUrl', '')[:60]}...")
        log(f"   - ldrUrl: {result.get('ldrUrl', '')[:60]}...")
        log(f"   - parts: {result.get('parts')}, finalTarget: {result.get('finalTarget')}")

        # ✅ RESULT 메시지 전송 (성공)
        log(f"📤 [SQS Consumer] RESULT 메시지 전송 중...")
        await send_result_message(
            job_id=job_id,
            success=True,
            corrected_url=result["correctedUrl"],
            glb_url=result["modelUrl"],
            ldr_url=result["ldrUrl"],
            bom_url=result["bomUrl"],
            parts=result["parts"],
            final_target=result["finalTarget"],
        )

        # ✅ 메시지 삭제 (처리 완료)
        delete_message(receipt_handle)

        log(f"✅ [SQS Consumer] REQUEST 메시지 처리 완료 | jobId={job_id}")
        log("=" * 60)

    except json.JSONDecodeError as e:
        log(f"❌ [SQS Consumer] JSON 파싱 실패 | messageId={message_id} | error={str(e)}")
        # 파싱 실패 메시지는 삭제 (재처리 불가)
        delete_message(receipt_handle)

    except Exception as e:
        log(f"❌ [SQS Consumer] 메시지 처리 실패 | messageId={message_id} | error={str(e)}")

        # ✅ RESULT 메시지 전송 (실패)
        try:
            job_id = json.loads(message["Body"]).get("jobId", "unknown")
            await send_result_message(
                job_id=job_id,
                success=False,
                error_message=str(e),
            )
        except Exception as send_error:
            log(f"❌ [SQS Consumer] 실패 메시지 전송 실패 | error={str(send_error)}")

        # AI 처리 실패 메시지는 삭제 (재처리 X, RESULT로 실패 전달함)
        delete_message(receipt_handle)


def delete_message(receipt_handle: str):
    """메시지 삭제"""
    try:
        sqs = _get_sqs_client()
        sqs.delete_message(
            QueueUrl=SQS_REQUEST_QUEUE_URL,
            ReceiptHandle=receipt_handle,
        )
    except Exception as e:
        log(f"❌ [SQS Consumer] 메시지 삭제 실패 | error={str(e)}")


async def start_consumer():
    """
    SQS Consumer 시작 (FastAPI startup event에서 호출)
    - 무한 루프로 폴링
    - 에러 발생 시에도 계속 실행
    """
    if not SQS_ENABLED:
        log("[SQS Consumer] ⚠️ SQS 비활성화 상태 (AWS_SQS_ENABLED=false)")
        return

    log("═" * 70)
    log("[SQS Consumer] 🚀 시작")
    log(f"   - Queue URL: {SQS_REQUEST_QUEUE_URL}")
    log(f"   - Poll Interval: {SQS_POLL_INTERVAL}초")
    log(f"   - Max Messages: {SQS_MAX_MESSAGES}")
    log(f"   - Wait Time: {SQS_WAIT_TIME}초 (Long polling)")
    log(f"   - Visibility Timeout: {SQS_VISIBILITY_TIMEOUT}초")
    log("═" * 70)

    while True:
        try:
            await poll_and_process()
            await asyncio.sleep(SQS_POLL_INTERVAL)

        except Exception as e:
            log(f"❌ [SQS Consumer] 예외 발생 | error={str(e)}")
            # 에러 발생해도 계속 실행
            await asyncio.sleep(SQS_POLL_INTERVAL)
