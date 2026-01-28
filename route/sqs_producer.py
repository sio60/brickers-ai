"""
SQS Producer - AI Server에서 Backend로 RESULT 메시지 전송
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional


def _is_truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


# 환경 변수
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2").strip()
SQS_QUEUE_URL = os.environ.get("AWS_SQS_QUEUE_URL", "").strip()
SQS_ENABLED = _is_truthy(os.environ.get("AWS_SQS_ENABLED", "false"))

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

    if not SQS_QUEUE_URL:
        raise RuntimeError("AWS_SQS_QUEUE_URL is not set")

    # boto3는 아래 env를 자동으로 읽음:
    # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
    _SQS_CLIENT = boto3.client("sqs", region_name=AWS_REGION)
    return _SQS_CLIENT


async def send_result_message(
    job_id: str,
    success: bool,
    corrected_url: str = "",
    glb_url: str = "",
    ldr_url: str = "",
    bom_url: str = "",
    parts: int = 0,
    final_target: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """
    RESULT 메시지를 Backend로 전송

    Args:
        job_id: Job ID
        success: 성공 여부
        corrected_url: 보정된 이미지 URL
        glb_url: GLB 파일 URL
        ldr_url: LDR 파일 URL
        bom_url: BOM 파일 URL
        parts: 파츠 수
        final_target: 최종 타겟
        error_message: 실패 시 에러 메시지
    """
    if not SQS_ENABLED:
        print(f"[SQS Producer] ⚠️ SQS 비활성화 상태 (메시지 전송 스킵) | jobId={job_id}")
        return

    try:
        client = _get_sqs_client()

        message = {
            "type": "RESULT",
            "jobId": job_id,
            "timestamp": datetime.now().isoformat(),
            "success": success,
        }

        if success:
            message.update({
                "correctedUrl": corrected_url,
                "glbUrl": glb_url,
                "ldrUrl": ldr_url,
                "bomUrl": bom_url,
                "parts": parts,
                "finalTarget": final_target,
            })
        else:
            message["errorMessage"] = error_message or "Unknown error"

        client.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps(message)
        )

        print(f"✅ [SQS Producer] RESULT 메시지 전송 완료 | jobId={job_id} | success={success}")

    except Exception as e:
        print(f"❌ [SQS Producer] 메시지 전송 실패 | jobId={job_id} | error={str(e)}")
        raise
