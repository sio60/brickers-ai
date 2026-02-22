# ============================================================================
# 인프라/운영 에이전트 도구 (스키마 + 실행 함수)
# Log Analyzer 에이전트가 서버 모니터링에 사용하는 도구들
# ============================================================================

import os
import logging
import psutil
from pydantic import BaseModel, Field

logger = logging.getLogger("agent.tools.infra")


# --- Pydantic 스키마 ---

class CheckDBStatus(BaseModel):
    """
    Checks the connection status and basic stats of the MongoDB database.
    Use this when logs indicate database timeouts or connection errors (e.g., ServerSelectionTimeoutError).
    """
    check_type: str = Field("ping", description="Type of check: 'ping' (connectivity) or 'stats' (collection counts). Default: 'ping'.")


class CheckSystemHealth(BaseModel):
    """
    Checks system resources (CPU, Memory, Disk).
    Use this when logs indicate 'MemoryError', 'Kill signal', or general slowness.
    """
    dummy: str = Field("ignore", description="Ignored field")


class ReadFileSnippet(BaseModel):
    """
    Reads a specific range of lines from a source code file.
    Use this to inspect code around an error traceback.
    """
    file_path: str = Field(..., description="Path to the file (e.g., 'route/kids_render.py'). Can be relative or absolute.")
    start_line: int = Field(1, description="Start line number (1-based, inclusive).")
    end_line: int = Field(..., description="End line number (1-based, inclusive). Limit to 50-100 lines at a time.")
    reasoning: str = Field(..., description="Why you need to read this file snippet.")


class CheckSQSStatus(BaseModel):
    """
    Checks the status of AWS SQS queues (Request/Result).
    Use this when logs indicate 'Boto3Error', 'Empty Message', or processing delays.
    """
    queue_type: str = Field("all", description="Which queue to check: 'request', 'result', or 'all'. Default: 'all'.")


# --- 실행 함수 ---

def execute_check_db(tool_input: dict) -> str:
    """MongoDB 연결 상태 확인"""
    logger.info(f"🔍 [Tool: check_db] Starting with input: {tool_input}")
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
        
        uri = os.getenv("MONGODB_URI")
        logger.info(f"📡 [check_db] Connecting to MongoDB (URI length: {len(uri) if uri else 0})")
        if not uri:
            logger.error("❌ [check_db] MONGODB_URI is missing!")
            return "Error: MONGODB_URI environment variable not set."

        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        
        if tool_input.get("check_type") == "stats":
            db_name = os.getenv("MONGODB_DB", "brickers_db")
            logger.info(f"📊 [check_db] Fetching stats for DB: {db_name}")
            db = client[db_name]
            stats = db.command("dbStats")
            res = f"MongoDB Status: Connected\nDB Name: {db_name}\nCollections: {stats.get('collections')}\nObjects: {stats.get('objects')}\nData Size: {stats.get('dataSize')} bytes"
            logger.info(f"✅ [check_db] Stats result: {stats.get('collections')} collections found.")
            return res
        else:
            client.admin.command('ping')
            logger.info("✅ [check_db] Ping success.")
            return "MongoDB Status: ✅ Connected (Ping successful)"

    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"❌ [check_db] Connection Failed: {str(e)}")
        return f"MongoDB Status: ❌ Connection Failed. Error: {str(e)}"
    except Exception as e:
        logger.error(f"❌ [check_db] Unexpected Error: {str(e)}")
        return f"MongoDB Status: ❌ Error: {str(e)}"


def execute_check_system(tool_input: dict) -> str:
    """시스템 리소스(CPU, 메모리, 디스크) 확인"""
    logger.info("🔍 [Tool: check_system] Starting resource check...")
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        res = f"""System Health:
        - CPU Usage: {cpu_percent}%
        - Memory: {memory.percent}% used (Available: {memory.available / (1024*1024):.0f} MB)
        - Disk: {disk.percent}% used (Free: {disk.free / (1024*1024*1024):.1f} GB)
        """
        logger.info(f"✅ [check_system] Result: CPU={cpu_percent}%, MEM={memory.percent}%")
        return res
    except Exception as e:
        logger.error(f"❌ [check_system] Failed: {str(e)}")
        return f"System Check Failed: {str(e)}"


def execute_read_file(tool_input: dict) -> str:
    """파일의 일부 내용을 읽습니다."""
    file_path = tool_input.get("file_path") or tool_input.get("filename")
    
    if not file_path:
        logger.error("❌ [read_file] 파일 경로(file_path)가 지정되지 않았습니다.")
        return "Error: file_path is required."

    logger.info(f"🔍 [도구: read_file] 요청 파일: {file_path}")
    try:
        start_line = tool_input.get("start_line", 1)
        end_line = tool_input.get("end_line", 100)
        
        if ".." in file_path:
             logger.warning(f"⚠️ [read_file] 보안 차단: {file_path}")
             return "Error: '..' 경로는 허용되지 않습니다."

        base_dir = "/app" if os.path.exists("/app") else os.getcwd()
        full_path = os.path.join(base_dir, file_path)
        logger.info(f"📂 [read_file] Full path resolved: {full_path}")
        
        if not os.path.exists(full_path):
            logger.error(f"❌ [read_file] File NOT found: {full_path}")
            return f"Error: File not found at {full_path}"

        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        start_index = max(0, start_line - 1)
        end_index = min(len(lines), end_line)
        
        snippet = "".join([f"{i+1}: {line}" for i, line in enumerate(lines[start_index:end_index], start=start_index)])
        logger.info(f"✅ [read_file] Read {end_index - start_index} lines.")
        return f"File: {file_path} ({start_line}-{end_index})\n\n{snippet}"

    except Exception as e:
        logger.error(f"❌ [read_file] Error: {str(e)}")
        return f"Error reading file: {str(e)}"


def execute_check_sqs(tool_input: dict) -> str:
    """AWS SQS 큐 상태 확인"""
    logger.info(f"🔍 [Tool: check_sqs] Starting with input: {tool_input}")
    try:
        import boto3
        
        region_name = os.getenv("AWS_REGION", "ap-northeast-2")
        aws_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if not aws_id or not aws_secret:
            logger.error("❌ [check_sqs] AWS Credentials missing in environment!")
            return "Error: AWS credentials not found in ENV."

        logger.info(f"📡 [check_sqs] Connecting to AWS SQS (Region: {region_name})")
        sqs = boto3.client(
            'sqs',
            region_name=region_name,
            aws_access_key_id=aws_id,
            aws_secret_access_key=aws_secret
        )
        
        request_url = os.getenv("AWS_SQS_REQUEST_QUEUE_URL")
        result_url = os.getenv("AWS_SQS_RESULT_QUEUE_URL")
        
        queues_to_check = {}
        target = tool_input.get("queue_type", "all")
        
        if target in ["request", "all"] and request_url:
            queues_to_check["RequestQueue"] = request_url
        if target in ["result", "all"] and result_url:
            queues_to_check["ResultQueue"] = result_url
            
        if not queues_to_check:
             logger.warning("⚠️ [check_sqs] No Queue URLs configured.")
             return "Warning: No SQS Queue URLs configured in environment."

        report = []
        for name, url in queues_to_check.items():
            logger.info(f"🚢 [check_sqs] Checking queue: {name}")
            try:
                response = sqs.get_queue_attributes(
                    QueueUrl=url,
                    AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
                )
                visible = response['Attributes']['ApproximateNumberOfMessages']
                inflight = response['Attributes']['ApproximateNumberOfMessagesNotVisible']
                report.append(f"- {name}: {visible} Waiting, {inflight} In-Flight")
                logger.info(f"✅ [check_sqs] {name}: {visible} messages waiting.")
            except Exception as qe:
                logger.error(f"❌ [check_sqs] {name} failed: {str(qe)}")
                report.append(f"- {name}: Check Failed ({str(qe)})")
                
        return "SQS Status:\n" + "\n".join(report)

    except Exception as e:
        logger.error(f"❌ [check_sqs] Critical Failure: {str(e)}")
        return f"SQS Tool Failed: {str(e)}"
