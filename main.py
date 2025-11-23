import logging
import os
import asyncio
import time
from datetime import datetime
from typing import Optional, Any
import uuid
from collections import deque
import re
import json

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI, OpenAIError, APITimeoutError

# 뉴스 크롤링용
import requests
from bs4 import BeautifulSoup

# 🆕 뉴스 필터링 시스템
try:
    from news_filter_simple import filter_real_estate_news, filter_news_batch
    NEWS_FILTER_AVAILABLE = True
except ImportError:
    NEWS_FILTER_AVAILABLE = False
    logging.warning("⚠️ news_filter_simple.py not found - filtering disabled")

# Redis for queue management
try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = Any
    logging.warning("redis package not installed. Using in-memory queue.")

# ================================================================================
# Logging Configuration
# ================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="부동산 뉴스 챗봇",
    description="Solar API + 부동산 뉴스 검색 챗봇",
    version="1.0.0"
)

# ================================================================================
# Configuration & Global Variables
# ================================================================================

# Naver News API
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Health Check Configuration
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", 5))
MAX_UNHEALTHY_COUNT = int(os.getenv("MAX_UNHEALTHY_COUNT", 3))

# Queue Configuration
WEBHOOK_QUEUE_NAME = "news_bot:webhook_queue"
WEBHOOK_PROCESSING_QUEUE = "news_bot:processing_queue"
WEBHOOK_FAILED_QUEUE = "news_bot:failed_queue"
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", 3))
QUEUE_PROCESS_INTERVAL = int(os.getenv("QUEUE_PROCESS_INTERVAL", 5))

# API Timeout Configuration
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 3))

# Global state
redis_client: Optional[Any] = None
server_healthy = True
unhealthy_count = 0
last_health_check = datetime.now()

# In-memory queue fallback
in_memory_webhook_queue: deque = deque()
in_memory_processing_queue: deque = deque()
in_memory_failed_queue: deque = deque()
use_in_memory_queue = False

# News session storage (user_id -> news_data)
news_sessions = {}

# ================================================================================
# Upstage Solar API Client (with timeout)
# ================================================================================

SOLAR_API_KEY = os.getenv("SOLAR_API_KEY", "")
SOLAR_API_BASE = "https://api.upstage.ai/v1/solar"

client = OpenAI(
    api_key=SOLAR_API_KEY,
    base_url=SOLAR_API_BASE,
    timeout=API_TIMEOUT
)

logger.info(f"✅ Solar API client initialized (Timeout: {API_TIMEOUT}s)")

# ================================================================================
# Naver News API - Search
# ================================================================================

def search_naver_news(query: str, display: int = 5):
    """네이버 뉴스 검색"""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        logger.warning("⚠️ Naver API credentials not configured")
        return []
    
    try:
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
        }
        params = {
            "query": query,
            "display": display,
            "sort": "date"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=3)
        response.raise_for_status()
        
        data = response.json()
        items = data.get("items", [])
        
        # HTML 태그 제거
        for item in items:
            item['title'] = re.sub(r'<[^>]+>', '', item['title'])
            item['description'] = re.sub(r'<[^>]+>', '', item['description'])
        
        logger.info(f"✅ Found {len(items)} news articles for: {query}")
        return items
        
    except Exception as e:
        logger.error(f"❌ Naver News API error: {e}")
        return []

def crawl_article(url: str) -> str:
    """기사 본문 크롤링"""
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 네이버 뉴스
        article = soup.find('article')
        if article:
            return article.get_text(strip=True)
        
        # 일반 기사
        paragraphs = soup.find_all('p')
        if paragraphs:
            return ' '.join([p.get_text(strip=True) for p in paragraphs[:10]])
        
        return ""
    except Exception as e:
        logger.error(f"❌ Crawl error: {e}")
        return ""

# ================================================================================
# Pydantic Models
# ================================================================================

class UserRequest(BaseModel):
    userRequest: dict
    bot: Optional[dict] = None
    action: Optional[dict] = None
    contexts: Optional[list] = None

class HealthStatus(BaseModel):
    status: str
    model: str
    mode: str
    server_healthy: bool
    last_check: str
    redis_connected: bool
    queue_size: int
    processing_queue_size: int
    failed_queue_size: int

class QueuedRequest(BaseModel):
    request_id: str
    payload: dict
    retry_count: int = 0
    created_at: datetime
    processing_started_at: Optional[datetime] = None

# ================================================================================
# Redis Queue Management
# ================================================================================

async def init_redis():
    """Initialize Redis connection with fallback to in-memory queue"""
    global redis_client, use_in_memory_queue
    
    if not REDIS_AVAILABLE:
        logger.warning("⚠️ Redis not installed - using in-memory queue")
        use_in_memory_queue = True
        return
    
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2
        )
        
        # Test connection
        await redis_client.ping()
        logger.info("✅ Redis connected successfully")
        use_in_memory_queue = False
        
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        logger.warning("⚠️ Using in-memory queue as fallback")
        redis_client = None
        use_in_memory_queue = True

async def close_redis():
    """Close Redis connection"""
    global redis_client
    
    if redis_client and not use_in_memory_queue:
        try:
            await redis_client.close()
            logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing Redis: {e}")
    
    redis_client = None

async def enqueue_webhook_request(request_id: str, payload: dict):
    """Enqueue failed webhook request for later processing"""
    queued_req = QueuedRequest(
        request_id=request_id,
        payload=payload,
        retry_count=0,
        created_at=datetime.now()
    )
    
    try:
        if use_in_memory_queue:
            in_memory_webhook_queue.append(queued_req)
            logger.info(f"✅ Request {request_id[:8]} enqueued (in-memory, size: {len(in_memory_webhook_queue)})")
            return
        
        if redis_client:
            await redis_client.lpush(WEBHOOK_QUEUE_NAME, queued_req.model_dump_json())
            queue_size = await redis_client.llen(WEBHOOK_QUEUE_NAME)
            logger.info(f"✅ Request {request_id[:8]} enqueued (Redis, size: {queue_size})")
        else:
            logger.warning("⚠️ No queue available - request lost")
            
    except Exception as e:
        logger.error(f"❌ Failed to enqueue request: {e}")

async def dequeue_webhook_request() -> Optional[QueuedRequest]:
    """Dequeue a webhook request from the queue"""
    try:
        if use_in_memory_queue:
            if len(in_memory_webhook_queue) > 0:
                return in_memory_webhook_queue.popleft()
            return None
        
        if redis_client:
            item = await redis_client.rpoplpush(WEBHOOK_QUEUE_NAME, WEBHOOK_PROCESSING_QUEUE)
            if item:
                return QueuedRequest.model_validate_json(item)
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Failed to dequeue request: {e}")
        return None

async def move_to_failed_queue(queued_req: QueuedRequest):
    """Move failed request to failed queue"""
    try:
        if use_in_memory_queue:
            in_memory_failed_queue.append(queued_req)
            logger.info(f"❌ Request {queued_req.request_id[:8]} moved to failed queue (in-memory)")
            return
        
        if redis_client:
            await redis_client.lpush(WEBHOOK_FAILED_QUEUE, queued_req.model_dump_json())
            await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, queued_req.model_dump_json())
            logger.info(f"❌ Request {queued_req.request_id[:8]} moved to failed queue (Redis)")
            
    except Exception as e:
        logger.error(f"❌ Failed to move request to failed queue: {e}")

async def complete_request(queued_req: QueuedRequest):
    """Remove completed request from processing queue"""
    try:
        if use_in_memory_queue:
            # Already removed from in-memory queue
            return
        
        if redis_client:
            await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, queued_req.model_dump_json())
            logger.info(f"✅ Request {queued_req.request_id[:8]} completed and removed from processing queue")
            
    except Exception as e:
        logger.error(f"❌ Failed to complete request: {e}")

async def get_queue_sizes():
    """Get current queue sizes"""
    try:
        if use_in_memory_queue:
            return (
                len(in_memory_webhook_queue),
                len(in_memory_processing_queue),
                len(in_memory_failed_queue)
            )
        
        if redis_client:
            webhook_size = await redis_client.llen(WEBHOOK_QUEUE_NAME)
            processing_size = await redis_client.llen(WEBHOOK_PROCESSING_QUEUE)
            failed_size = await redis_client.llen(WEBHOOK_FAILED_QUEUE)
            return (webhook_size, processing_size, failed_size)
        
        return (0, 0, 0)
        
    except Exception as e:
        logger.error(f"❌ Failed to get queue sizes: {e}")
        return (0, 0, 0)

# ================================================================================
# Background Tasks
# ================================================================================

async def queue_processor():
    """Background task to process queued webhook requests"""
    logger.info("🔄 Queue processor started")
    
    while True:
        try:
            await asyncio.sleep(QUEUE_PROCESS_INTERVAL)
            
            queued_req = await dequeue_webhook_request()
            if not queued_req:
                continue
            
            logger.info(f"⚙️ Processing queued request {queued_req.request_id[:8]} (retry: {queued_req.retry_count})")
            
            try:
                result = await process_chatbot_request(queued_req.payload)
                await complete_request(queued_req)
                logger.info(f"✅ Queued request {queued_req.request_id[:8]} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to process queued request: {e}")
                queued_req.retry_count += 1
                
                if queued_req.retry_count >= MAX_RETRY_ATTEMPTS:
                    await move_to_failed_queue(queued_req)
                else:
                    await enqueue_webhook_request(queued_req.request_id, queued_req.payload)
                    
        except Exception as e:
            logger.error(f"❌ Queue processor error: {e}")
            await asyncio.sleep(5)

async def health_check_monitor():
    """Background task to monitor server health"""
    global server_healthy, unhealthy_count, last_health_check
    
    logger.info("❤️ Health check monitor started")
    
    while True:
        try:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            
            # Simple health check - check if Solar API is responsive
            try:
                test_response = client.chat.completions.create(
                    model="solar-mini",
                    messages=[{"role": "user", "content": "health check"}],
                    max_tokens=5
                )
                
                if test_response.choices[0].message.content:
                    server_healthy = True
                    unhealthy_count = 0
                    last_health_check = datetime.now()
                else:
                    raise Exception("Empty response from Solar API")
                    
            except Exception as e:
                unhealthy_count += 1
                logger.error(f"❌ Health check failed ({unhealthy_count}/{MAX_UNHEALTHY_COUNT}): {e}")
                
                if unhealthy_count >= MAX_UNHEALTHY_COUNT:
                    server_healthy = False
                    logger.error(f"❌ Server unhealthy: {unhealthy_count} consecutive failures")
                    
        except Exception as e:
            logger.error(f"❌ Health check monitor error: {e}")
            await asyncio.sleep(5)

# ================================================================================
# Core Processing Logic
# ================================================================================

async def process_chatbot_request(payload: dict):
    """
    Solar API + 뉴스 검색 처리 로직
    """
    try:
        user_msg = payload.get("userRequest", {}).get("utterance", "")
        user_id = payload.get("userRequest", {}).get("user", {}).get("id", "unknown")
        
        logger.info(f"👤 User {user_id[:8]}: {user_msg}")
        
        # ================================================================================
        # 1) 뉴스 세션 관리
        # ================================================================================
        if user_id in news_sessions:
            news_data = news_sessions[user_id]
            
            # 번호 선택 처리
            if user_msg.strip().isdigit():
                choice = int(user_msg.strip())
                if 1 <= choice <= len(news_data):
                    selected = news_data[choice - 1]
                    article_content = crawl_article(selected['link'])
                    
                    if not article_content:
                        answer_text = "기사 본문을 가져올 수 없습니다."
                    else:
                        qa_prompt = f"""다음 부동산 기사를 읽고 핵심 내용을 2-3문장으로 요약해주세요.

기사 제목: {selected['title']}
기사 내용: {article_content[:1500]}

답변:"""
                        
                        response = client.chat.completions.create(
                            model="solar-mini",
                            messages=[{"role": "user", "content": qa_prompt}],
                            max_tokens=150
                        )
                        answer_text = response.choices[0].message.content.strip()
                    
                    del news_sessions[user_id]
                    
                    return {
                        "version": "2.0",
                        "template": {
                            "outputs": [
                                {"simpleText": {"text": answer_text}}
                            ]
                        }
                    }
            
            # 취소
            if "취소" in user_msg or "그만" in user_msg:
                del news_sessions[user_id]
                return {
                    "version": "2.0",
                    "template": {
                        "outputs": [
                            {"simpleText": {"text": "뉴스 검색을 취소했습니다."}}
                        ]
                    }
                }
        
        # ================================================================================
        # 2) 뉴스 검색 키워드 감지
        # ================================================================================
        news_keywords = ["뉴스", "기사", "최신", "소식"]
        if any(kw in user_msg for kw in news_keywords):
            search_query = user_msg
            for kw in news_keywords:
                search_query = search_query.replace(kw, "").strip()
            
            if not search_query:
                search_query = "부동산"
            
            logger.info(f"📰 Searching news for: {search_query}")
            news_items = search_naver_news(search_query, display=10)
            
            if not news_items:
                return {
                    "version": "2.0",
                    "template": {
                        "outputs": [
                            {"simpleText": {"text": "검색 결과가 없습니다."}}
                        ]
                    }
                }
            
            # 🆕 뉴스 필터링 적용
            if NEWS_FILTER_AVAILABLE:
                logger.info(f"🔍 Filtering {len(news_items)} news articles...")
                filtered_items = filter_news_batch(news_items)
                logger.info(f"✅ Filtered to {len(filtered_items)} relevant articles")
                
                if not filtered_items:
                    return {
                        "version": "2.0",
                        "template": {
                            "outputs": [
                                {"simpleText": {"text": "부동산 관련 뉴스가 없습니다."}}
                            ]
                        }
                    }
                
                news_items = filtered_items[:5]
            else:
                news_items = news_items[:5]
            
            news_sessions[user_id] = news_items
            
            news_list = "\n".join([
                f"{i+1}. {item['title']}"
                for i, item in enumerate(news_items)
            ])
            
            answer_text = f"검색 결과:\n\n{news_list}\n\n번호를 입력하시면 상세 내용을 알려드립니다."
            
            return {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {"simpleText": {"text": answer_text}}
                    ]
                }
            }
        
        # ================================================================================
        # 3) 일반 대화
        # ================================================================================
        response = client.chat.completions.create(
            model="solar-mini",
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=200
        )
        
        answer_text = response.choices[0].message.content.strip()
        
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": answer_text}}
                ]
            }
        }
    
    except APITimeoutError:
        logger.warning(f"⏰ Solar API timeout after {API_TIMEOUT}s")
        raise
    except OpenAIError as e:
        logger.error(f"❌ Solar API error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {type(e).__name__}: {e}")
        raise

# ================================================================================
# FastAPI Endpoints
# ================================================================================

@app.get("/")
async def root():
    """루트 엔드포인트 - Render 헬스 체크용"""
    return {
        "service": "부동산 뉴스 챗봇",
        "version": "1.0.0",
        "status": "running",
        "healthy": server_healthy,
        "timestamp": datetime.now().isoformat(),
        "features": {
            "news": NAVER_CLIENT_ID is not None,
            "filtering": NEWS_FILTER_AVAILABLE,
            "redis": redis_client is not None and not use_in_memory_queue
        }
    }

@app.post("/webhook/chatbot")
async def webhook_endpoint(request: UserRequest):
    """
    카카오톡 챗봇 웹훅 엔드포인트
    """
    request_id = str(uuid.uuid4())
    
    logger.info("="*50)
    logger.info(f"📨 New request received: {request_id[:8]}")
    logger.info(f"📋 Full request body: {request.model_dump()}")
    
    try:
        # 3초 타임아웃으로 빠른 응답 시도
        result = await process_chatbot_request(request.model_dump())
        logger.info(f"✅ Request {request_id[:8]} completed successfully")
        return result
        
    except APITimeoutError as e:
        logger.warning(f"⏰ Timeout (3s) - enqueueing request {request_id}")
        await enqueue_webhook_request(request_id, request.model_dump())
        
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "답변 생성에 시간이 걸리고 있습니다. 잠시 후 다시 질문해주세요."
                        }
                    }
                ]
            }
        }
        
    except OpenAIError as e:
        logger.error(f"❌ API Error: {e}")
        await enqueue_webhook_request(request_id, request.model_dump())
        
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
                        }
                    }
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {type(e).__name__}: {e}")
        await enqueue_webhook_request(request_id, request.model_dump())
        
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "죄송합니다. 오류가 발생했습니다. 다시 한번 질문해주시겠어요?"
                        }
                    }
                ]
            }
        }

@app.get("/health")
async def health_check() -> HealthStatus:
    """Enhanced health check endpoint"""
    queue_size, processing_size, failed_size = await get_queue_sizes()
    
    return HealthStatus(
        status="healthy" if server_healthy else "unhealthy",
        model="solar-mini",
        mode="news_chatbot",
        server_healthy=server_healthy,
        last_check=last_health_check.isoformat(),
        redis_connected=(redis_client is not None and not use_in_memory_queue),
        queue_size=queue_size,
        processing_queue_size=processing_size,
        failed_queue_size=failed_size
    )

@app.get("/health/ping")
async def health_ping():
    """Simple ping endpoint for client health checks"""
    return {
        "alive": True,
        "healthy": server_healthy,
        "timestamp": datetime.now().isoformat(),
        "news_sessions": len(news_sessions)
    }

@app.get("/queue/status")
async def queue_status():
    """Get detailed queue status"""
    queue_size, processing_size, failed_size = await get_queue_sizes()
    
    return {
        "queue_type": "in-memory" if use_in_memory_queue else "redis",
        "webhook_queue": queue_size,
        "processing_queue": processing_size,
        "failed_queue": failed_size,
        "total": queue_size + processing_size + failed_size,
        "active_news_sessions": len(news_sessions)
    }

@app.post("/queue/retry-failed")
async def retry_failed_requests():
    """Manually retry all failed requests"""
    try:
        if use_in_memory_queue:
            retry_count = len(in_memory_failed_queue)
            while len(in_memory_failed_queue) > 0:
                req = in_memory_failed_queue.pop()
                req.retry_count = 0
                in_memory_webhook_queue.appendleft(req)
            
            logger.info(f"✅ Retrying {retry_count} failed requests (in-memory)")
            return {"retried": retry_count, "queue_type": "in-memory"}
        
        if not redis_client:
            return {"error": "Queue not available"}
        
        failed_items = await redis_client.lrange(WEBHOOK_FAILED_QUEUE, 0, -1)
        retry_count = 0
        
        for item in failed_items:
            req = QueuedRequest.model_validate_json(item)
            req.retry_count = 0
            await redis_client.lpush(WEBHOOK_QUEUE_NAME, req.model_dump_json())
            retry_count += 1
        
        await redis_client.delete(WEBHOOK_FAILED_QUEUE)
        
        logger.info(f"✅ Retrying {retry_count} failed requests (Redis)")
        return {"retried": retry_count, "queue_type": "redis"}
        
    except Exception as e:
        logger.error(f"❌ Failed to retry requests: {e}")
        return {"error": str(e)}

# ================================================================================
# Startup & Shutdown Events
# ================================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("="*70)
    logger.info("🚀 Starting 부동산 뉴스 챗봇...")
    logger.info("="*70)
    
    # Naver API 확인
    if NAVER_CLIENT_ID and NAVER_CLIENT_SECRET:
        logger.info("✅ Naver News API configured")
    else:
        logger.warning("⚠️ Naver News API not configured")
    
    # 🆕 News Filtering 확인
    if NEWS_FILTER_AVAILABLE:
        logger.info("✅ News filtering system enabled")
    else:
        logger.warning("⚠️ News filtering system disabled")
        logger.warning("   Place news_filter_simple.py in the same directory")
    
    # Redis 초기화
    await init_redis()
    
    # Background tasks
    asyncio.create_task(health_check_monitor())
    asyncio.create_task(queue_processor())
    
    logger.info("="*70)
    logger.info("✅ 부동산 뉴스 챗봇 시작 완료!")
    logger.info(f"   - Model: solar-mini")
    logger.info(f"   - Redis: {'connected' if redis_client else 'in-memory queue'}")
    logger.info(f"   - News API: {'enabled' if NAVER_CLIENT_ID else 'disabled'}")
    logger.info(f"   - News Filter: {'enabled' if NEWS_FILTER_AVAILABLE else 'disabled'}")
    logger.info("="*70)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("👋 Shutting down 부동산 뉴스 챗봇...")
    await close_redis()
    logger.info("✅ Server shut down successfully")
