"""
Gallery Screenshot Backfill Script
기존 갤러리 포스트 중 ldrUrl이 있고 screenshotUrls가 없는 것들을 찾아
Celery 태스크로 스크린샷 생성을 요청합니다.

사용법:
  # 대상 확인만 (dry-run, 기본값)
  python backfill.py

  # 실제 전송
  python backfill.py --send

필수 환경변수:
  MONGODB_URI                    - MongoDB Atlas 연결 URI
  MONGODB_DB                     - 데이터베이스명 (예: brickers)
  CELERY_SQS_SCREENSHOT_QUEUE_URL - Celery SQS 큐 URL
  AWS_REGION                     - AWS 리전 (기본: ap-northeast-2)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

try:
    from pymongo import MongoClient
    from celery import Celery
except ImportError:
    print("필수 패키지 설치: pip install pymongo celery")
    sys.exit(1)


def _get_celery_app():
    queue_url = os.environ.get("CELERY_SQS_SCREENSHOT_QUEUE_URL", "").strip()
    region = os.environ.get("AWS_REGION", "ap-northeast-2").strip()
    if not queue_url:
        print("ERROR: CELERY_SQS_SCREENSHOT_QUEUE_URL 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    app = Celery("backfill_producer")
    app.conf.update(
        broker_url="sqs://",
        broker_transport_options={
            "region": region,
            "predefined_queues": {
                "celery-screenshots": {"url": queue_url},
            },
            "queue_name_prefix": "",
        },
        task_serializer="json",
        accept_content=["json"],
        result_backend=None,
        task_default_queue="celery-screenshots",
    )
    return app


def main():
    dry_run = "--send" not in sys.argv

    mongo_uri = os.environ.get("MONGODB_URI", "").strip()
    mongo_db = os.environ.get("MONGODB_DB", "brickers").strip()

    if not mongo_uri:
        print("ERROR: MONGODB_URI 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    print(f"MongoDB 연결 중... (db={mongo_db})")
    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    collection = db["gallery_posts"]

    query = {
        "deleted": False,
        "ldrUrl": {"$ne": None, "$exists": True, "$not": {"$eq": ""}},
        "$or": [
            {"screenshotUrls": None},
            {"screenshotUrls": {"$exists": False}},
            {"screenshotUrls": {}},
        ],
    }

    posts = list(collection.find(query, {"_id": 1, "title": 1, "ldrUrl": 1}))
    print(f"\n백필 대상: {len(posts)}개 갤러리 포스트\n")

    if not posts:
        print("백필 대상이 없습니다.")
        return

    for i, post in enumerate(posts, 1):
        post_id = str(post["_id"])
        title = post.get("title", "untitled")
        ldr_url = post["ldrUrl"]
        print(f"  [{i:3d}] id={post_id} | title={title[:30]} | ldr={ldr_url[:60]}...")

    if dry_run:
        print(f"\nDRY RUN 모드 - 전송하지 않음")
        print(f"   실제 전송하려면: python backfill.py --send")
        return

    celery_app = _get_celery_app()
    print(f"\nCelery 태스크 전송 시작...")
    sent = 0
    failed = 0

    for i, post in enumerate(posts, 1):
        post_id = str(post["_id"])
        title = post.get("title", "untitled")
        ldr_url = post["ldrUrl"]

        try:
            celery_app.send_task(
                "tasks.process_screenshot",
                kwargs={
                    "job_id": post_id,
                    "ldr_url": ldr_url,
                    "model_name": title,
                    "source": "gallery_backfill",
                    "gallery_post_id": post_id,
                },
                queue="celery-screenshots",
            )
            sent += 1
            print(f"  [{i:3d}] 전송 | id={post_id} | {title[:30]}")
        except Exception as e:
            failed += 1
            print(f"  [{i:3d}] 실패 | id={post_id} | error={str(e)}")

    print(f"\n{'=' * 50}")
    print(f"결과: 전송 {sent}개 | 실패 {failed}개 | 전체 {len(posts)}개")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
