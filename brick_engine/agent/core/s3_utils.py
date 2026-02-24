# ============================================================================
# S3 유틸리티 (S3 Utils)
# S3 버킷과의 상호작용(업로드/다운로드)을 위한 간단한 유틸리티입니다.
# ============================================================================

import boto3
import logging
from pathlib import Path

logger = logging.getLogger("agent.core.s3_utils")

def download_from_s3(bucket: str, key: str, local_path: str):
    """S3에서 파일을 다운로드합니다."""
    try:
        s3 = boto3.client('s3')
        s3.download_file(bucket, key, local_path)
        logger.info(f"Successfully downloaded s3://{bucket}/{key} to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        return False

def upload_to_s3(local_path: str, bucket: str, key: str):
    """로컬 파일을 S3에 업로드합니다."""
    try:
        s3 = boto3.client('s3')
        s3.upload_file(local_path, bucket, key)
        logger.info(f"Successfully uploaded {local_path} to s3://{bucket}/{key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False
