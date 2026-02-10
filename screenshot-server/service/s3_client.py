# screenshot-server/service/s3_client.py
"""S3 업로드 유틸리티 (경량화 버전 - Screenshot 서버 전용)"""
from __future__ import annotations

import os

# boto3 lazy import
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore


def _is_truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


AWS_REGION = (
    os.environ.get("AWS_REGION", "").strip()
    or os.environ.get("AWS_DEFAULT_REGION", "").strip()
)

S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "").strip()
S3_PUBLIC_BASE_URL = os.environ.get("S3_PUBLIC_BASE_URL", "").strip().rstrip("/")
USE_S3 = _is_truthy(os.environ.get("USE_S3", "true" if S3_BUCKET else "false"))

_S3_CLIENT = None


def _require_s3_ready() -> None:
    if not USE_S3:
        return
    if boto3 is None:
        raise RuntimeError("boto3 is not installed (pip install boto3)")
    if not S3_BUCKET:
        raise RuntimeError("AWS_S3_BUCKET is not set")
    if not AWS_REGION:
        raise RuntimeError("AWS_REGION is not set")


def get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    _require_s3_ready()
    _S3_CLIENT = boto3.client("s3", region_name=AWS_REGION)
    return _S3_CLIENT


def public_s3_url(key: str) -> str:
    if S3_PUBLIC_BASE_URL:
        return f"{S3_PUBLIC_BASE_URL}/{key}"
    if AWS_REGION == "us-east-1":
        return f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "image/png") -> str:
    """bytes 데이터를 S3에 업로드하고 URL 반환."""
    if not USE_S3:
        return ""
    client = get_s3_client()
    if not client:
        return ""
    extra_args = {
        "ContentType": content_type,
    }
    client.put_object(Bucket=S3_BUCKET, Key=key, Body=data, **extra_args)
    return public_s3_url(key)
