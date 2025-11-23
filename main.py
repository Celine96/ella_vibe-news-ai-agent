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
import numpy as np
import pickle

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
    title="REXA - Real Estate Expert Assistant",
    description="Solar API + RAG chatbot for real estate + News QA",
    version="2.0.0"
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
WEBHOOK_QUEUE_NAME = "rexa:webhook_queue"
WEBHOOK_PROCESSING_QUEUE = "rexa:processing_queue"
WEBHOOK_FAILED_QUEUE = "rexa:failed_queue"
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
# RAG System - Load embeddings and chunks
# ================================================================================

# 임베딩 및 청크 데이터 로드 (전역)
try:
    with open("chunk_embeddings.pkl", "rb") as f:
        chunk_embeddings = pickle.load(f)
    logger.info(f"✅ Loaded {len(chunk_embeddings)} chunk embeddings")
except FileNotFoundError:
    chunk_embeddings = []
    logger.warning("⚠️ chunk_embeddings.pkl not found - RAG disabled")

try:
    with open("article_chunks.pkl", "rb") as f:
        article_chunks = pickle.load(f)
    logger.info(f"✅ Loaded {len(article_chunks)} article chunks")
except FileNotFoundError:
    article_chunks = []
    logger.warning("⚠️ article_chunks.pkl not found - RAG disabled")

def retrieve_relevant_chunks(question: str, top_k: int = 3):
    """
    질문 임베딩을 생성하고, 코사인 유사도로 상위 k개 청크 반환
    """
    if len(chunk_embeddings) == 0:
        logger.warning("⚠️ No embeddings loaded - RAG retrieve failed")
        return []
    
    try:
        # Solar embedding 모델 사용
        response = client.embeddings.create(
            model="solar-embedding-1-large-query",
            input=question
        )
        q_emb = np.array(response.data[0].embedding)
        
        # 코사인 유사도
        scores = []
        for i, c_emb in enumerate(chunk_embeddings):
            sim = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb) + 1e-9)
            scores.append((i, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scores[:top_k]]
        return [article_chunks[i] for i in top_indices]
    
    except Exception as e:
        logger.error(f"❌ RAG retrieve error: {e}")
        return []

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
        redis_client = await redis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
            password=REDIS_PASSWORD,
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping()
        logger.info(f"✅ Redis connected: {REDIS_HOST}:{REDIS_PORT}")
        use_in_memory_queue = False
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        logger.warning("⚠️ Using in-memory queue as fallback")
        redis_client = None
        use_in_memory_queue = True

async def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("✅ Redis connection closed")

async def enqueue_webhook_request(request_id: str, payload: dict):
    """Enqueue a webhook request for later processing"""
    request = QueuedRequest(
        request_id=request_id,
        payload=payload,
        created_at=datetime.now()
    )
    
    if use_in_memory_queue:
        in_memory_webhook_queue.append(request)
        logger.info(f"✅ Request {request_id[:8]} enqueued (in-memory, size: {len(in_memory_webhook_queue)})")
        return
    
    if not redis_client:
        logger.error("❌ Queue not available")
        return
    
    try:
        await redis_client.lpush(WEBHOOK_QUEUE_NAME, request.model_dump_json())
        queue_size = await redis_client.llen(WEBHOOK_QUEUE_NAME)
        logger.info(f"✅ Request {request_id[:8]} enqueued (Redis, size: {queue_size})")
    except Exception as e:
        logger.error(f"❌ Failed to enqueue request: {e}")

async def dequeue_webhook_request() -> Optional[QueuedRequest]:
    """Dequeue and process next webhook request"""
    if use_in_memory_queue:
        if len(in_memory_webhook_queue) == 0:
            return None
        request = in_memory_webhook_queue.popleft()
        in_memory_processing_queue.append(request)
        return request
    
    if not redis_client:
        return None
    
    try:
        # Move from webhook queue to processing queue
        data = await redis_client.rpoplpush(WEBHOOK_QUEUE_NAME, WEBHOOK_PROCESSING_QUEUE)
        if not data:
            return None
        
        request = QueuedRequest.model_validate_json(data)
        request.processing_started_at = datetime.now()
        return request
    except Exception as e:
        logger.error(f"❌ Failed to dequeue request: {e}")
        return None

async def mark_request_completed(request: QueuedRequest):
    """Remove request from processing queue after successful completion"""
    if use_in_memory_queue:
        try:
            in_memory_processing_queue.remove(request)
        except ValueError:
            pass
        return
    
    if not redis_client:
        return
    
    try:
        await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, request.model_dump_json())
    except Exception as e:
        logger.error(f"❌ Failed to mark request as completed: {e}")

async def mark_request_failed(request: QueuedRequest):
    """Move request to failed queue after max retries"""
    if use_in_memory_queue:
        try:
            in_memory_processing_queue.remove(request)
        except ValueError:
            pass
        in_memory_failed_queue.append(request)
        return
    
    if not redis_client:
        return
    
    try:
        await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, request.model_dump_json())
        await redis_client.lpush(WEBHOOK_FAILED_QUEUE, request.model_dump_json())
    except Exception as e:
        logger.error(f"❌ Failed to mark request as failed: {e}")

async def requeue_request(request: QueuedRequest):
    """Put request back in queue for retry"""
    request.retry_count += 1
    
    if use_in_memory_queue:
        try:
            in_memory_processing_queue.remove(request)
        except ValueError:
            pass
        in_memory_webhook_queue.appendleft(request)
        return
    
    if not redis_client:
        return
    
    try:
        await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, request.model_dump_json())
        await redis_client.lpush(WEBHOOK_QUEUE_NAME, request.model_dump_json())
    except Exception as e:
        logger.error(f"❌ Failed to requeue request: {e}")

async def get_queue_sizes() -> tuple[int, int, int]:
    """Get sizes of all queues"""
    if use_in_memory_queue:
        return (
            len(in_memory_webhook_queue),
            len(in_memory_processing_queue),
            len(in_memory_failed_queue)
        )
    
    if not redis_client:
        return (0, 0, 0)
    
    try:
        queue_size = await redis_client.llen(WEBHOOK_QUEUE_NAME)
        processing_size = await redis_client.llen(WEBHOOK_PROCESSING_QUEUE)
        failed_size = await redis_client.llen(WEBHOOK_FAILED_QUEUE)
        return (queue_size, processing_size, failed_size)
    except Exception as e:
        logger.error(f"❌ Failed to get queue sizes: {e}")
        return (0, 0, 0)

# ================================================================================
# Background Workers
# ================================================================================

async def health_check_monitor():
    """Monitor server health"""
    global server_healthy, unhealthy_count, last_health_check
    
    while True:
        try:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            
            # Check if we can still make API calls
            try:
                test_response = client.chat.completions.create(
                    model="solar-mini",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5,
                    timeout=2
                )
                server_healthy = True
                unhealthy_count = 0
            except Exception as e:
                unhealthy_count += 1
                if unhealthy_count >= MAX_UNHEALTHY_COUNT:
                    server_healthy = False
                    logger.error(f"❌ Server unhealthy: {unhealthy_count} consecutive failures")
            
            last_health_check = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Health check monitor error: {e}")

async def queue_processor():
    """Background worker to process queued requests"""
    logger.info("🔄 Queue processor started")
    
    while True:
        try:
            await asyncio.sleep(QUEUE_PROCESS_INTERVAL)
            
            request = await dequeue_webhook_request()
            if not request:
                continue
            
            logger.info(f"🔄 Processing queued request {request.request_id[:8]} (attempt {request.retry_count + 1})")
            
            try:
                # Process the request
                result = await process_solar_rag_request(request.payload)
                await mark_request_completed(request)
                logger.info(f"✅ Queued request {request.request_id[:8]} completed")
                
            except Exception as e:
                logger.error(f"❌ Queued request {request.request_id[:8]} failed: {e}")
                
                if request.retry_count < MAX_RETRY_ATTEMPTS - 1:
                    await requeue_request(request)
                    logger.info(f"🔄 Requeued request {request.request_id[:8]} (retry {request.retry_count + 1})")
                else:
                    await mark_request_failed(request)
                    logger.error(f"❌ Request {request.request_id[:8]} moved to failed queue")
        
        except Exception as e:
            logger.error(f"❌ Queue processor error: {e}")

# ================================================================================
# Main Logic - Process Solar RAG Request
# ================================================================================

async def process_solar_rag_request(request_body: dict):
    """
    실제 요청 처리: Solar + RAG + 뉴스
    """
    utterance_raw = request_body.get("userRequest", {}).get("utterance", "").strip()
    user_id = request_body.get("userRequest", {}).get("user", {}).get("id", "unknown")
    
    logger.info(f"👤 User: {user_id}")
    logger.info(f"💬 Question: {utterance_raw}")
    
    # === 뉴스 답변 모드 체크 ===
    if user_id in news_sessions:
        news_data = news_sessions[user_id]
        question = utterance_raw.lower()
        
        # 뉴스 번호 추출 (1-5)
        match = re.search(r'(\d+)', question)
        if match:
            news_idx = int(match.group(1)) - 1
            if 0 <= news_idx < len(news_data):
                news = news_data[news_idx]
                
                # 링크 크롤링
                article_text = crawl_article(news['link'])
                if not article_text:
                    article_text = news['description']
                
                # GPT로 답변 생성
                prompt = f"""다음은 부동산 뉴스 기사입니다:

제목: {news['title']}
내용: {article_text}

사용자 질문: {utterance_raw}

위 기사 내용을 바탕으로 사용자의 질문에 답변해주세요. 답변은 친절하고 전문적으로 작성하되, 200자 이내로 간결하게 해주세요."""

                try:
                    response = client.chat.completions.create(
                        model="solar-mini",
                        messages=[{"role": "user", "content": prompt}],
                        timeout=API_TIMEOUT
                    )
                    answer = response.choices[0].message.content.strip()
                    
                    # 세션 종료
                    del news_sessions[user_id]
                    
                    return {
                        "version": "2.0",
                        "template": {
                            "outputs": [
                                {"simpleText": {"text": answer}}
                            ]
                        }
                    }
                except Exception as e:
                    logger.error(f"❌ GPT answer error: {e}")
                    del news_sessions[user_id]
                    return {
                        "version": "2.0",
                        "template": {
                            "outputs": [
                                {"simpleText": {"text": "기사 분석 중 오류가 발생했습니다."}}
                            ]
                        }
                    }
        
        # 잘못된 입력
        del news_sessions[user_id]
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": "올바른 번호를 입력해주세요 (예: 1번)"}}
                ]
            }
        }
    
    # === 뉴스 검색 요청 처리 ===
    news_keywords = ["뉴스", "최근", "기사", "소식"]
    if any(kw in utterance_raw for kw in news_keywords):
        # 검색어 추출
        query = utterance_raw
        for kw in news_keywords:
            query = query.replace(kw, "").strip()
        
        if not query:
            query = "부동산"
        
        # 네이버 뉴스 검색
        news_items = search_naver_news(query, display=10)
        
        if not news_items:
            return {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {"simpleText": {"text": f"'{query}' 관련 뉴스를 찾을 수 없습니다."}}
                    ]
                }
            }
        
        # 🆕 뉴스 필터링 (GPT-4o-mini)
        if NEWS_FILTER_AVAILABLE:
            filtered_news = filter_news_batch(news_items)
            # 관련도 높은 순으로 정렬
            filtered_news.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            # 상위 5개만
            filtered_news = filtered_news[:5]
        else:
            # 필터링 없이 상위 5개
            filtered_news = news_items[:5]
        
        if not filtered_news:
            return {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {"simpleText": {"text": f"'{query}' 관련 부동산 뉴스를 찾을 수 없습니다."}}
                    ]
                }
            }
        
        # 뉴스 세션 저장
        news_sessions[user_id] = filtered_news
        
        # 뉴스 목록 텍스트 생성
        news_text = f"'{query}' 관련 부동산 뉴스 {len(filtered_news)}건을 찾았습니다:\n\n"
        for i, news in enumerate(filtered_news, 1):
            relevance = news.get('relevance_score', 0)
            keywords_str = ', '.join(news.get('keywords', [])[:3]) if news.get('keywords') else ''
            
            news_text += f"{i}. {news['title']}\n"
            if keywords_str:
                news_text += f"   키워드: {keywords_str}\n"
            if NEWS_FILTER_AVAILABLE and relevance > 0:
                news_text += f"   관련도: {relevance}점\n"
            news_text += "\n"
        
        news_text += "자세히 알고 싶은 뉴스 번호를 입력해주세요 (예: 1번)"
        
        logger.info(f"✅ Found {len(filtered_news)} relevant news articles")
        
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": news_text}}
                ]
            }
        }
    
    # === 일반 질문 (RAG + Solar) ===
    rag_context = ""
    if len(chunk_embeddings) > 0:
        top_chunks = retrieve_relevant_chunks(utterance_raw, top_k=3)
        if top_chunks:
            rag_context = "\n\n".join(top_chunks)
            logger.info(f"✅ RAG: Retrieved {len(top_chunks)} chunks")
    
    system_msg = """당신은 REXA, 부동산 전문 AI 어시스턴트입니다.
금하빌딩(서울 강남구 논현동 21-1)에 대한 전문 지식을 보유하고 있으며, 
상업용 부동산 임대차보호법, 양도소득세, 시세 분석 등을 안내합니다.

답변 원칙:
1. 제공된 문서 내용을 우선 활용
2. 전문적이면서도 친근한 톤
3. 구체적 수치나 법률은 정확하게
4. 200자 내외로 간결하게"""

    if rag_context:
        user_msg = f"""[참고 문서]
{rag_context}

[사용자 질문]
{utterance_raw}"""
    else:
        user_msg = utterance_raw
    
    try:
        response = client.chat.completions.create(
            model="solar-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            timeout=API_TIMEOUT
        )
        
        answer_text = response.choices[0].message.content.strip()
        logger.info(f"✅ Solar response generated")
        
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
# FastAPI Endpoints
# ================================================================================

@app.post("/webhook/solar-rag")
async def webhook_endpoint(request: UserRequest):
    """
    카카오톡 챗봇 웹훅 엔드포인트
    """
    request_id = str(uuid.uuid4())
    
    logger.info("="*50)
    logger.info(f"📨 New RAG request received: {request_id[:8]}")
    logger.info(f"📋 Full request body: {request.model_dump()}")
    
    try:
        # 3초 타임아웃으로 빠른 응답 시도
        result = await process_solar_rag_request(request.model_dump())
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
        mode="rexa_chatbot_rag_news",
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
        "rag_enabled": len(chunk_embeddings) > 0,
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
        "rag_chunks_loaded": len(article_chunks),
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
    logger.info("🚀 Starting REXA server (Solar + RAG + News + Filtering)...")
    logger.info("="*70)
    
    # RAG 상태 확인
    if len(chunk_embeddings) > 0:
        logger.info(f"✅ RAG ENABLED: {len(chunk_embeddings)} chunks loaded")
    else:
        logger.warning("⚠️ RAG DISABLED: No embeddings loaded")
        logger.warning("⚠️ Server will work but without company-specific knowledge")
    
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
    logger.info("✅ REXA server startup complete!")
    logger.info(f"   - Model: solar-mini")
    logger.info(f"   - RAG chunks: {len(chunk_embeddings)}")
    logger.info(f"   - Redis: {'connected' if redis_client else 'in-memory queue'}")
    logger.info(f"   - News API: {'enabled' if NAVER_CLIENT_ID else 'disabled'}")
    logger.info(f"   - News Filter: {'enabled' if NEWS_FILTER_AVAILABLE else 'disabled'}")
    logger.info("="*70)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("👋 Shutting down REXA server (Solar + RAG + News)...")
    await close_redis()
    logger.info("✅ REXA server shut down successfully")
