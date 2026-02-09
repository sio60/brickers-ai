import os
import logging
import psutil
from typing import List
from pydantic import BaseModel, Field

# Logger configuration
logger = logging.getLogger("agent.tools")


# ============================================================================
# 에이전트 도구 스키마 정의
# LLM이 Function Calling으로 사용할 수 있는 도구들의 입출력 형식 정의
#
# NOTE: LLM은 검증/분석만 수행하고, LDR 직접 수정은 하지 않음.
#       개선은 알고리즘 재실행(TuneParameters)을 통해서만 이루어짐.
# ============================================================================

class TuneParameters(BaseModel):
    """
    구조물 안정성을 개선하기 위해 GLB-to-LDR 변환 파라미터를 조정합니다.
    이전 시도 결과를 바탕으로 새로운 파라미터 조합을 제안해야 합니다.
    """
    target: int = Field(..., description="목표 스터드 크기 (기본값: 25). 크기가 클수록 디테일이 살지만 불안정할 수 있음.")
    budget: int = Field(..., description="최대 브릭 사용 개수 (기본값: 150).")
    interlock: bool = Field(..., description="인터락(엇갈려 쌓기) 활성화 여부. 안정성을 위해 필수적임.")
    fill: bool = Field(..., description="내부 채움 활성화 여부. 끄면 속이 비어 가벼워지지만 약해짐.")
    smart_fix: bool = Field(..., description="스마트 보정 활성화 여부.")
    plates_per_voxel: int = Field(..., description="복셀당 플레이트 수 (1~3). 3이면 정밀하지만 브릭 수가 늘어남.")
    auto_remove_1x1: bool = Field(..., description="True면 1x1 브릭을 자동 삭제하여 안정성을 확보합니다. 디테일이 중요하다면 False로 설정하세요.")
    support_ratio: float = Field(..., description="지지 비율(0~1). 높을수록 안정적이지만 브릭 수가 증가합니다.")
    small_side_contact: bool = Field(..., description="작은 브릭의 사이드 접촉 허용 여부.")
    reasoning: str = Field(..., description="이 파라미터를 선택한 이유에 대한 간략한 설명.")


class RemoveBricks(BaseModel):
    """
    특정 브릭들을 삭제하여 안정성을 확보합니다.
    주의: 점수가 90점 이상이고, 소수의 공중부양(Floating) 브릭만 문제일 때 사용하세요.
    """
    brick_ids: List[str] = Field(..., description="삭제할 브릭 ID 목록 (예: ['3005.dat_0', '3024.dat_5'])")
    reasoning: str = Field(..., description="삭제 이유 (예: '점수 92점이나 공중부양 브릭 2개 발생하여 제거')")


# --- Infrastructure Tools ---

class CheckDBStatus(BaseModel):
    """
    Checks the connection status and basic stats of the MongoDB database.
    Use this when logs indicate database timeouts or connection errors (e.g., ServerSelectionTimeoutError).
    """
    check_type: str = Field("ping", description="Type of check: 'ping' (connectivity) or 'stats' (collection counts). Default: 'ping'.")

def execute_check_db(tool_input: dict) -> str:
    """Executes MongoDB check."""
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
            # Ping
            client.admin.command('ping')
            logger.info("✅ [check_db] Ping success.")
            return "MongoDB Status: ✅ Connected (Ping successful)"

    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"❌ [check_db] Connection Failed: {str(e)}")
        return f"MongoDB Status: ❌ Connection Failed. Error: {str(e)}"
    except Exception as e:
        logger.error(f"❌ [check_db] Unexpected Error: {str(e)}")
        return f"MongoDB Status: ❌ Error: {str(e)}"

class CheckSystemHealth(BaseModel):
    """
    Checks system resources (CPU, Memory, Disk).
    Use this when logs indicate 'MemoryError', 'Kill signal', or general slowness.
    """
    dummy: str = Field("ignore", description="Ignored field")

def execute_check_system(tool_input: dict) -> str:
    """Executes system health check."""
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

class ReadFileSnippet(BaseModel):
    """
    Reads a specific range of lines from a source code file.
    Use this to inspect code around an error traceback.
    """
    file_path: str = Field(..., description="Path to the file (e.g., 'route/kids_render.py'). Can be relative or absolute.")
    start_line: int = Field(1, description="Start line number (1-based, inclusive).")
    end_line: int = Field(..., description="End line number (1-based, inclusive). Limit to 50-100 lines at a time.")
    reasoning: str = Field(..., description="Why you need to read this file snippet.")

def execute_read_file(tool_input: dict) -> str:
    """파일의 일부 내용을 읽습니다."""
    # AI가 'filename'으로 보낼 수도 있으므로 유연하게 처리
    file_path = tool_input.get("file_path") or tool_input.get("filename")
    
    if not file_path:
        logger.error("❌ [read_file] 파일 경로(file_path)가 지정되지 않았습니다.")
        return "Error: file_path is required."

    logger.info(f"🔍 [도구: read_file] 요청 파일: {file_path}")
    try:
        start_line = tool_input.get("start_line", 1)
        end_line = tool_input.get("end_line", 100)
        
        # 보안 체크 (기본)
        if ".." in file_path:
             logger.warning(f"⚠️ [read_file] 보안 차단: {file_path}")
             return "Error: '..' 경로는 허용되지 않습니다."

        # Determine Base Directory (Docker vs Local)
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

class CheckSQSStatus(BaseModel):
    """
    Checks the status of AWS SQS queues (Request/Result).
    Use this when logs indicate 'Boto3Error', 'Empty Message', or processing delays.
    """
    queue_type: str = Field("all", description="Which queue to check: 'request', 'result', or 'all'. Default: 'all'.")

def execute_check_sqs(tool_input: dict) -> str:
    """Executes SQS status check."""
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
        
        # Load Queue URLs from ENV
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
