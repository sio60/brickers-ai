# service/s3_client.py
"""S3 업로드 / 다운로드 / URL 생성 유틸리티"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from service.kids_config import _is_truthy, GENERATED_DIR, STATIC_PREFIX

# boto3 lazy import
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore

AI_PUBLIC_BASE_URL = os.environ.get("AI_PUBLIC_BASE_URL", "").strip().rstrip("/")

AWS_REGION = (
    os.environ.get("AWS_REGION", "").strip()
    or os.environ.get("AWS_DEFAULT_REGION", "").strip()
)

S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "").strip()
S3_PUBLIC_BASE_URL = os.environ.get("S3_PUBLIC_BASE_URL", "").strip().rstrip("/")
USE_S3 = _is_truthy(os.environ.get("USE_S3", "true" if S3_BUCKET else "false"))
S3_PREFIX = os.environ.get("S3_PREFIX", "uploads/ai-generated").strip().strip("/")
S3_PRESIGN = _is_truthy(os.environ.get("S3_PRESIGN", "false"))
S3_PRESIGN_EXPIRES = int(os.environ.get("S3_PRESIGN_EXPIRES", "86400"))
S3_FORCE_ATTACHMENT = _is_truthy(os.environ.get("S3_FORCE_ATTACHMENT", "true"))
S3_USE_ACL_PUBLIC_READ = _is_truthy(os.environ.get("S3_USE_ACL_PUBLIC_READ", "false"))

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


def presigned_get_url(key: str) -> str:
    client = get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=S3_PRESIGN_EXPIRES,
    )


def s3_url_for_key(key: str) -> str:
    if S3_PRESIGN:
        return presigned_get_url(key)
    return public_s3_url(key)


def upload_to_s3(local_path: Path, key: str, content_type: str | None = None) -> str:
    """로컬 파일을 S3에 업로드하고 URL 반환. USE_S3=false면 빈 문자열."""
    if not USE_S3:
        return ""
    client = get_s3_client()
    extra_args: dict = {}
    if content_type:
        extra_args["ContentType"] = content_type
    if S3_FORCE_ATTACHMENT and local_path.suffix.lower() in (".ldr", ".glb"):
        extra_args["ContentDisposition"] = f'attachment; filename="{local_path.name}"'
    if S3_USE_ACL_PUBLIC_READ:
        extra_args["ACL"] = "public-read"
    client.upload_file(str(local_path), S3_BUCKET, key, ExtraArgs=extra_args if extra_args else None)
    return s3_url_for_key(key)


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "application/pdf") -> str:
    """bytes 데이터를 S3에 업로드하고 URL 반환."""
    if not USE_S3:
        return ""
    client = get_s3_client()
    if not client:
        return ""
    extra_args = {
        "ContentType": content_type,
        "ContentDisposition": f'attachment; filename="{key.split("/")[-1]}"',
    }
    client.put_object(Bucket=S3_BUCKET, Key=key, Body=data, **extra_args)
    return s3_url_for_key(key)


def guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".glb": "application/octet-stream",
        ".gltf": "model/gltf+json",
        ".ldr": "text/plain",
        ".json": "application/json",
    }.get(ext, "application/octet-stream")


def to_generated_url(p: Path, out_dir: Path) -> str:
    """
    USE_S3=true: S3에 업로드 후 S3 URL 반환
    USE_S3=false: GENERATED_DIR 기준 /api/generated/... URL 반환
    """
    p = Path(p).resolve()
    gen = GENERATED_DIR.resolve()

    # GENERATED_DIR 밖 파일이면 out_dir로 복사
    try:
        rel = p.relative_to(gen)
    except ValueError:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / p.name
        if p != dst:
            dst.write_bytes(p.read_bytes())
        p = dst
        rel = dst.relative_to(gen)

    if USE_S3:
        try:
            now = datetime.now()
            year, month = now.year, now.month
            s3_key = f"{S3_PREFIX}/{year:04d}/{month:02d}/{rel.as_posix()}" if S3_PREFIX else rel.as_posix()
            content_type = guess_content_type(p)
            s3_url = upload_to_s3(p, s3_key, content_type)
            if s3_url:
                return s3_url
        except Exception as e:
            print(f"[S3 upload failed, fallback to local] {e}")

    url = f"{STATIC_PREFIX}/" + rel.as_posix()
    if AI_PUBLIC_BASE_URL:
        return f"{AI_PUBLIC_BASE_URL}{url}"
    return url
