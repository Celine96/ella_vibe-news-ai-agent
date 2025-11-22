import logging
import os
import asyncio
import time
from datetime import datetime
from typing import Optional, Any
import uuid
from collections import deque
import re
import csv
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

# Google Sheets용
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    logging.warning("gspread not installed. Google Sheets logging disabled.")

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

# Google Sheets Configuration
GOOGLE_SHEETS_CREDENTIALS = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
GOOGLE_SHEETS_SPREADSHEET_ID = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")

# CSV 파일 경로
CSV_FILE_PATH = "news_log.csv"

# Google Sheets 클라이언트 (전역)
gsheet_client = None
gsheet_worksheet = None

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
# Google Sheets & CSV Initialization
# ================================================================================

def init_google_sheets():
    """Initialize Google Sheets client"""
    global gsheet_client, gsheet_worksheet
    
    if not GSPREAD_AVAILABLE:
        logger.error("❌ gspread not installed - Google Sheets logging disabled")
        logger.error("   Install: pip install gspread google-auth --break-system-packages")
        return False
    
    if not GOOGLE_SHEETS_CREDENTIALS:
        logger.error("❌ GOOGLE_SHEETS_CREDENTIALS environment variable not set")
        logger.error("   Set in Render: Environment → GOOGLE_SHEETS_CREDENTIALS")
        return False
    
    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        logger.error("❌ GOOGLE_SHEETS_SPREADSHEET_ID environment variable not set")
        logger.error("   Set in Render: Environment → GOOGLE_SHEETS_SPREADSHEET_ID")
        return False
    
    try:
        logger.info("🔄 Initializing Google Sheets...")
        
        # JSON 문자열을 딕셔너리로 파싱
        try:
            creds_dict = json.loads(GOOGLE_SHEETS_CREDENTIALS)
            logger.info("✅ Credentials JSON parsed successfully")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in GOOGLE_SHEETS_CREDENTIALS: {e}")
            return False
        
        # Credentials 생성
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        logger.info("✅ Google credentials created")
        
        # gspread 클라이언트 생성
        gsheet_client = gspread.authorize(credentials)
        logger.info("✅ gspread client authorized")
        
        # 스프레드시트 열기
        spreadsheet = gsheet_client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
        gsheet_worksheet = spreadsheet.sheet1
        logger.info(f"✅ Spreadsheet opened: {spreadsheet.title}")
        
        # 헤더 확인 및 생성
        try:
            headers = gsheet_worksheet.row_values(1)
            if not headers or headers[0] != 'timestamp':
                # 🆕 새로운 컬럼 구조
                gsheet_worksheet.insert_row([
                    'timestamp', 'title', 'description', 'url',
                    'is_relevant', 'relevance_score', 'keywords', 'region',
                    'has_price', 'has_policy', 'reason', 'user_id'
                ], 1)
                logger.info("✅ Google Sheets headers created (with filtering columns)")
            else:
                logger.info(f"✅ Google Sheets headers found: {headers}")
        except Exception as e:
            # 🆕 새로운 컬럼 구조
            gsheet_worksheet.insert_row([
                'timestamp', 'title', 'description', 'url',
                'is_relevant', 'relevance_score', 'keywords', 'region',
                'has_price', 'has_policy', 'reason', 'user_id'
            ], 1)
            logger.info("✅ Google Sheets headers created (with filtering columns)")
        
        logger.info(f"✅ Google Sheets initialized: {GOOGLE_SHEETS_SPREADSHEET_ID}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Sheets: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False

def init_csv_file():
    """Initialize CSV file with headers"""
    try:
        # 파일이 없으면 헤더 생성
        if not os.path.exists(CSV_FILE_PATH):
            with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 🆕 새로운 컬럼 구조
                writer.writerow([
                    'timestamp', 'title', 'description', 'url',
                    'is_relevant', 'relevance_score', 'keywords', 'region',
                    'has_price', 'has_policy', 'reason', 'user_id'
                ])
            logger.info(f"✅ CSV file created: {CSV_FILE_PATH}")
        else:
            logger.info(f"✅ CSV file exists: {CSV_FILE_PATH}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize CSV: {e}")
        return False

def save_news_to_csv(news_data: dict):
    """Save news to CSV file with filtering metadata"""
    try:
        with open(CSV_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 🆕 새로운 컬럼 구조
            writer.writerow([
                news_data['timestamp'],
                news_data['title'],
                news_data['description'],
                news_data['url'],
                news_data.get('is_relevant', True),
                news_data.get('relevance_score', 0),
                ', '.join(news_data.get('keywords', [])),
                news_data.get('region', ''),
                news_data.get('has_price', False),
                news_data.get('has_policy', False),
                news_data.get('reason', ''),
                news_data['user_id']
            ])
        logger.info(f"✅ News saved to CSV: {news_data['title'][:30]}...")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save to CSV: {e}")
        return False

def save_news_to_gsheet(news_data: dict):
    """Save news to Google Sheets with filtering metadata"""
    if not gsheet_worksheet:
        logger.warning("⚠️ Google Sheets not initialized - skipping")
        return False
    
    try:
        # 🆕 새로운 컬럼 구조
        gsheet_worksheet.append_row([
            news_data['timestamp'],
            news_data['title'],
            news_data['description'],
            news_data['url'],
            news_data.get('is_relevant', True),
            news_data.get('relevance_score', 0),
            ', '.join(news_data.get('keywords', [])),
            news_data.get('region', ''),
            news_data.get('has_price', False),
            news_data.get('has_policy', False),
            news_data.get('reason', ''),
            news_data['user_id']
        ])
        logger.info(f"✅ News saved to Google Sheets: {news_data['title'][:30]}...")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save to Google Sheets: {e}")
        return False

def save_news_log(title: str, description: str, url: str, content: str = "", user_id: str = "unknown"):
    """Save news to both CSV and Google Sheets"""
    news_data = {
        'timestamp': datetime.now().isoformat(),
        'title': title,
        'description': description,
        'url': url,
        'content': content,
        'user_id': user_id
    }
    
    # CSV 저장 (백업)
    save_news_to_csv(news_data)
    
    # Google Sheets 저장 (메인)
    save_news_to_gsheet(news_data)


# ================================================================================
# Upstage Solar API Configuration
# ================================================================================

client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1/solar",
    timeout=API_TIMEOUT
)

logger.info("✅ Upstage Solar API client configured")

# ================================================================================
# RAG - Load Embeddings
# ================================================================================

article_chunks = []
chunk_embeddings = []

try:
    with open("embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        article_chunks = data["chunks"]
        chunk_embeddings = data["embeddings"]
    logger.info(f"✅ Loaded {len(article_chunks)} chunks from embeddings.pkl")
    logger.info(f"✅ RAG is ENABLED with {len(article_chunks)} chunks")
except FileNotFoundError:
    logger.warning("⚠️ embeddings.pkl not found - RAG will not be available")
    logger.warning("⚠️ Server will continue WITHOUT RAG - responses will be general")
    logger.warning("⚠️ To enable RAG: run 'python embedding2_solar.py' and redeploy")
except Exception as e:
    logger.error(f"❌ Failed to load embeddings: {e}")
    logger.warning("⚠️ Server will continue WITHOUT RAG")

# ================================================================================
# News Functions
# ================================================================================

def search_naver_news(query: str = "부동산", display: int = 10) -> Optional[list]:
    """네이버 뉴스 API로 최신 뉴스 검색 + 부동산 관련성 필터링"""
    url = "https://openapi.naver.com/v1/search/news.json"
    
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    params = {
        "query": query,
        "display": display,
        "sort": "date"  # 최신순
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        items = data.get('items', [])
        if not items:
            return None
        
        # 네이버 뉴스 도메인만 필터링
        naver_items = [item for item in items if 'news.naver.com' in item['link']]
        
        if not naver_items:
            logger.warning("⚠️ 네이버 뉴스가 없습니다. 일반 뉴스를 사용합니다.")
            naver_items = items  # 폴백: 모든 뉴스 사용
        
        logger.info(f"✅ 네이버 뉴스 {len(naver_items)}개 발견")
        
        # 모든 뉴스 아이템 처리
        processed_items = []
        for item in naver_items:
            # HTML 태그 제거
            title = re.sub('<[^<]+?>', '', item['title'])
            description = re.sub('<[^<]+?>', '', item['description'])
            
            # HTML 엔티티 디코딩
            import html
            title = html.unescape(title)
            description = html.unescape(description)
            
            # 요약 길이 제한 (200자, 문장 단위로)
            if len(description) > 200:
                cut_pos = 200
                for i in range(200, max(0, len(description) - 100), -1):
                    if description[i] in '.!?':
                        cut_pos = i + 1
                        break
                description = description[:cut_pos].strip()
            
            processed_items.append({
                "title": title,
                "description": description,
                "link": item['link'],
                "pubDate": item['pubDate'],
                "timestamp": datetime.now().isoformat()
            })
        
        # 🆕 부동산 관련성 필터링
        if NEWS_FILTER_AVAILABLE:
            logger.info(f"🔍 필터링 시작: {len(processed_items)}개 기사")
            filtered_items = filter_news_batch(processed_items)
            logger.info(
                f"✅ 필터링 완료: {len(processed_items)}개 중 {len(filtered_items)}개 관련 기사 "
                f"({len(filtered_items)/len(processed_items)*100:.1f}%)"
            )
            return filtered_items
        else:
            logger.warning("⚠️ 필터링 모듈 없음 - 모든 기사 반환")
            return processed_items
        
    except Exception as e:
        logger.error(f"❌ 뉴스 검색 오류: {e}")
        return None

def crawl_news_content(url: str) -> str:
    """뉴스 URL에서 본문 추출 - 전체 원문 (재시도 포함)"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://news.naver.com/',
                'Connection': 'keep-alive'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 네이버 뉴스 본문 추출
            if 'news.naver.com' in url:
                article = soup.select_one('#dic_area') or soup.select_one('#articeBody') or soup.select_one('.news_end')
                if article:
                    # 불필요한 태그 제거
                    for tag in article.find_all(['script', 'style', 'aside']):
                        tag.decompose()
                    content = article.get_text(strip=True, separator='\n')
                    logger.info(f"📄 크롤링 성공: {len(content)}자")
                    return content  # 전체 원문 반환 (제한 없음)
            
            # 일반 뉴스 사이트 - p 태그 기반 추출
            paragraphs = soup.find_all('p')
            content = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
            
            if content:
                logger.info(f"📄 크롤링 성공: {len(content)}자")
                return content  # 전체 원문 반환
            else:
                return "본문을 추출할 수 없습니다."
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ 타임아웃 발생 - 재시도 {attempt + 1}/{max_retries}")
                time.sleep(2)  # 2초 대기 후 재시도
                continue
            else:
                logger.error(f"❌ 크롤링 타임아웃: {url[:50]}...")
                return "본문을 가져올 수 없습니다. (타임아웃)"
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Too Many Requests
                if attempt < max_retries - 1:
                    wait_time = 3  # 3초 대기
                    logger.warning(f"⚠️ Rate Limit (429) - {wait_time}초 대기 후 재시도")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ Rate Limit 초과: {url[:50]}...")
                    return "본문을 가져올 수 없습니다. (Rate Limit)"
            else:
                logger.error(f"❌ HTTP 오류 {e.response.status_code}: {url[:50]}...")
                return f"본문을 가져올 수 없습니다. (HTTP {e.response.status_code})"
        except Exception as e:
            logger.error(f"❌ 크롤링 오류: {e}")
            return "본문을 가져올 수 없습니다."
    
    return "본문을 가져올 수 없습니다."
# ================================================================================
# RAG Helper Functions
# ================================================================================

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    from numpy import dot
    from numpy.linalg import norm
    return dot(a, b) / (norm(a) * norm(b))

async def get_relevant_context(prompt: str, top_n: int = 2) -> str:
    """Get relevant context from embeddings for RAG"""
    if not chunk_embeddings or not article_chunks:
        logger.warning("⚠️ No embeddings available for RAG")
        return ""
    
    try:
        # 임베딩 차원 자동 감지
        embedding_dim = len(chunk_embeddings[0])
        logger.info(f"📊 Detected embedding dimension: {embedding_dim}")
        
        # 차원에 따라 적절한 API 사용
        if embedding_dim == 1536:
            # OpenAI 임베딩 (text-embedding-3-small)
            logger.info("🔧 Using OpenAI embedding model")
            try:
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                q_embedding = openai_client.embeddings.create(
                    input=prompt, 
                    model="text-embedding-3-small"
                ).data[0].embedding
            except Exception as e:
                logger.error(f"❌ OpenAI embedding failed: {e}")
                logger.info("💡 Set OPENAI_API_KEY environment variable")
                return ""
                
        else:
            # Solar 임베딩 (모든 다른 차원)
            logger.info(f"🔧 Using Solar embedding model (dimension: {embedding_dim})")
            try:
                q_embedding = client.embeddings.create(
                    input=prompt, 
                    model="solar-embedding-1-large-query"  # Solar 쿼리용 모델
                ).data[0].embedding
            except Exception as e:
                logger.error(f"❌ Solar embedding failed: {e}")
                logger.error(f"   Model: solar-embedding-1-large-query")
                return ""
        
        # Calculate similarities
        similarities = [cosine_similarity(q_embedding, emb) for emb in chunk_embeddings]
        
        # Get top N most similar chunks
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        selected_context = "\n\n".join([article_chunks[i] for i in top_indices])
        
        # Format similarities for logging
        similarity_scores = [f"{similarities[i]:.3f}" for i in top_indices]
        logger.info(f"✅ Retrieved {top_n} relevant chunks (similarities: {similarity_scores})")
        return selected_context
        
    except Exception as e:
        logger.error(f"❌ Error getting relevant context: {e}")
        return ""

# ================================================================================
# Pydantic Models
# ================================================================================

class DetailParams(BaseModel):
    prompt: dict

class Action(BaseModel):
    params: dict
    detailParams: dict

class RequestBody(BaseModel):
    action: Action

class QueuedRequest(BaseModel):
    request_id: str
    request_body: dict
    timestamp: str
    retry_count: int = 0
    error_message: Optional[str] = None

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

# ================================================================================
# Redis & Queue Management
# ================================================================================

async def init_redis():
    """Initialize Redis connection"""
    global redis_client, use_in_memory_queue
    
    if not REDIS_AVAILABLE:
        logger.warning("⚠️ Redis package not installed - using in-memory queue")
        use_in_memory_queue = True
        return
    
    try:
        redis_client = await redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            retry_on_timeout=True
        )
        await redis_client.ping()
        logger.info(f"✅ Redis connected: {REDIS_HOST}:{REDIS_PORT}")
        use_in_memory_queue = False
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        logger.warning("⚠️ Using in-memory queue as fallback")
        use_in_memory_queue = True

async def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("✅ Redis connection closed")

async def enqueue_webhook_request(request_id: str, request_body: dict):
    """Add webhook request to queue"""
    queued_req = QueuedRequest(
        request_id=request_id,
        request_body=request_body,
        timestamp=datetime.now().isoformat()
    )
    
    if use_in_memory_queue:
        in_memory_webhook_queue.append(queued_req)
        logger.info(f"📥 Request {request_id[:8]} added to in-memory queue (size: {len(in_memory_webhook_queue)})")
        return
    
    if not redis_client:
        in_memory_webhook_queue.append(queued_req)
        logger.warning(f"⚠️ Redis unavailable - using in-memory queue")
        return
    
    try:
        await redis_client.lpush(WEBHOOK_QUEUE_NAME, queued_req.model_dump_json())
        queue_size = await redis_client.llen(WEBHOOK_QUEUE_NAME)
        logger.info(f"📥 Request {request_id[:8]} added to Redis queue (size: {queue_size})")
    except Exception as e:
        logger.error(f"❌ Failed to enqueue to Redis: {e}")
        in_memory_webhook_queue.append(queued_req)
        logger.info(f"📥 Fallback to in-memory queue (size: {len(in_memory_webhook_queue)})")

async def get_queue_sizes():
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
        webhook_size = await redis_client.llen(WEBHOOK_QUEUE_NAME)
        processing_size = await redis_client.llen(WEBHOOK_PROCESSING_QUEUE)
        failed_size = await redis_client.llen(WEBHOOK_FAILED_QUEUE)
        return (webhook_size, processing_size, failed_size)
    except Exception as e:
        logger.error(f"❌ Failed to get queue sizes: {e}")
        return (0, 0, 0)

async def health_check_monitor():
    """Background task to monitor server health"""
    global server_healthy, unhealthy_count, last_health_check
    
    while True:
        try:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            
            # Check Solar API
            try:
                test_response = client.chat.completions.create(
                    model="solar-mini",
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=10,
                    timeout=2
                )
                
                server_healthy = True
                unhealthy_count = 0
                last_health_check = datetime.now()
                logger.debug(f"✅ Health check passed at {last_health_check}")
                
            except Exception as e:
                unhealthy_count += 1
                logger.warning(f"⚠️ Health check failed ({unhealthy_count}/{MAX_UNHEALTHY_COUNT}): {e}")
                
                if unhealthy_count >= MAX_UNHEALTHY_COUNT:
                    server_healthy = False
                    logger.error(f"❌ Server marked as unhealthy after {unhealthy_count} failures")
                
        except Exception as e:
            logger.error(f"❌ Health check monitor error: {e}")

async def queue_processor():
    """Background task to process queued requests"""
    while True:
        try:
            await asyncio.sleep(QUEUE_PROCESS_INTERVAL)
            
            if use_in_memory_queue:
                while len(in_memory_webhook_queue) > 0:
                    req = in_memory_webhook_queue.popleft()
                    
                    try:
                        result = await process_solar_rag_request(req.request_body)
                        logger.info(f"✅ Processed queued request {req.request_id[:8]}")
                    except Exception as e:
                        req.retry_count += 1
                        req.error_message = str(e)
                        
                        if req.retry_count < MAX_RETRY_ATTEMPTS:
                            in_memory_webhook_queue.append(req)
                            logger.warning(f"⚠️ Retry {req.retry_count}/{MAX_RETRY_ATTEMPTS} for {req.request_id[:8]}")
                        else:
                            in_memory_failed_queue.append(req)
                            logger.error(f"❌ Request {req.request_id[:8]} moved to failed queue")
                continue
            
            if not redis_client:
                continue
            
            # Process from Redis queue
            req_json = await redis_client.rpoplpush(WEBHOOK_QUEUE_NAME, WEBHOOK_PROCESSING_QUEUE)
            
            if not req_json:
                continue
            
            req = QueuedRequest.model_validate_json(req_json)
            
            try:
                result = await process_solar_rag_request(req.request_body)
                await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, req_json)
                logger.info(f"✅ Processed queued request {req.request_id[:8]}")
                
            except Exception as e:
                req.retry_count += 1
                req.error_message = str(e)
                
                await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, req_json)
                
                if req.retry_count < MAX_RETRY_ATTEMPTS:
                    await redis_client.lpush(WEBHOOK_QUEUE_NAME, req.model_dump_json())
                    logger.warning(f"⚠️ Retry {req.retry_count}/{MAX_RETRY_ATTEMPTS} for {req.request_id[:8]}")
                else:
                    await redis_client.lpush(WEBHOOK_FAILED_QUEUE, req.model_dump_json())
                    logger.error(f"❌ Request {req.request_id[:8]} moved to failed queue")
                    
        except Exception as e:
            logger.error(f"❌ Queue processor error: {e}")

# ================================================================================
# Core Processing Functions
# ================================================================================

async def process_solar_rag_request(request_body: dict) -> dict:
    """Process request with Solar API + RAG or News context"""
    try:
        action = request_body.get("action", {})
        detail_params = action.get("detailParams", {})
        user_message_data = detail_params.get("prompt", {})
        user_message = user_message_data.get("value", "").strip()
        
        # 사용자 ID 추출 (카카오톡 userRequest에서)
        user_request = request_body.get("userRequest", {})
        user_info = user_request.get("user", {})
        user_id = user_info.get("id", "default")
        
        if not user_message:
            return {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {"simpleText": {"text": "질문을 입력해주세요."}}
                    ]
                }
            }
        
        logger.info(f"💬 User message: {user_message}")
        
        # 컨텍스트 결정: 뉴스 세션이 있으면 뉴스, 없으면 RAG
        context = ""
        context_source = "general"
        
        # 뉴스 세션 확인
        if user_id in news_sessions:
            news_data = news_sessions[user_id]
            context = f"다음은 최신 부동산 뉴스입니다:\n\n제목: {news_data['title']}\n\n{news_data['content']}\n\n위 뉴스를 참고하여 사용자의 질문에 답변해주세요."
            context_source = "news"
            logger.info(f"📰 Using news context for user {user_id}")
        else:
            # RAG 컨텍스트 사용
            context = await get_relevant_context(user_message, top_n=2)
            if context:
                context = f"다음은 관련 정보입니다:\n\n{context}\n\n위 정보를 참고하여 답변해주세요."
                context_source = "rag"
                logger.info(f"📚 Using RAG context")
        
        # System prompt 구성
        system_prompt = "당신은 부동산 전문 AI 어시스턴트 REXA입니다."
        if context:
            system_prompt += f"\n\n{context}"
        
        # Solar API 호출
        response = client.chat.completions.create(
            model="solar-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            timeout=API_TIMEOUT
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        logger.info(f"✅ Solar API response received (context: {context_source})")
        logger.info(f"📝 Response: {ai_response[:100]}...")
        
        # 뉴스 모드일 경우 Quick Reply 추가
        kakao_response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": ai_response}}
                ]
            }
        }
        
        if context_source == "news":
            news_data = news_sessions[user_id]
            kakao_response["template"]["quickReplies"] = [
                {
                    "label": "뉴스 원문 보기",
                    "action": "webLink",
                    "webLinkUrl": news_data['url']
                },
                {
                    "label": "다른 질문하기",
                    "action": "message",
                    "messageText": "이 뉴스에서 핵심은 뭐야?"
                }
            ]
        
        return kakao_response
        
    except Exception as e:
        logger.error(f"❌ Solar API error: {type(e).__name__}: {e}")
        raise

# ================================================================================
# API Endpoints
# ================================================================================

@app.post("/generate")
async def generate(request: RequestBody):
    """REXA 부동산 전문 챗봇 with RAG - 카카오톡 5초 제한 대응"""
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

@app.post("/news")
async def news_bot(request: RequestBody):
    """부동산 뉴스봇 - 즉시 응답 후 백그라운드에서 저장"""
    request_id = str(uuid.uuid4())
    
    logger.info("="*50)
    logger.info(f"📰 News bot request received: {request_id[:8]}")
    
    try:
        # 사용자 ID 추출
        request_dict = request.model_dump()
        user_request = request_dict.get("userRequest", {})
        user_info = user_request.get("user", {})
        user_id = user_info.get("id", "default")
        
        # 네이버 뉴스 검색 (5개)
        news_items = search_naver_news("부동산", display=5)
        
        if not news_items or len(news_items) == 0:
            return {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {"simpleText": {"text": "뉴스를 불러올 수 없습니다. 잠시 후 다시 시도해주세요."}}
                    ]
                }
            }
        
        logger.info(f"📊 총 {len(news_items)}개 뉴스 발견")
        
        # 첫 번째 뉴스만 즉시 크롤링 (사용자 응답용)
        first_news = news_items[0]
        first_news_content = crawl_news_content(first_news['link'])
        
        # 세션에 저장 (질의응답용)
        news_sessions[user_id] = {
            "title": first_news['title'],
            "description": first_news['description'],
            "content": first_news_content,
            "url": first_news['link'],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ 첫 번째 뉴스: {first_news['title'][:50]}...")
        
        # Solar AI 요약 생성
        try:
            summary_prompt = f"다음 뉴스를 3-4개의 완전한 문장으로 요약해주세요. 문장 중간에 끊기지 않도록 주의하세요.\n\n제목: {first_news['title']}\n\n본문: {first_news_content[:1500]}"
            
            response = client.chat.completions.create(
                model="solar-mini",
                messages=[
                    {"role": "system", "content": "당신은 부동산 뉴스 전문 요약 AI입니다. 항상 완전한 문장으로 요약합니다."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=300,
                timeout=API_TIMEOUT
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"✅ Solar AI summary generated")
            
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            summary = first_news['description']
            last_period = summary.rfind('.')
            if last_period > 0:
                summary = summary[:last_period + 1]
        
        # 백그라운드 작업: 모든 뉴스 저장 (비동기)
        asyncio.create_task(save_all_news_background(news_items, user_id))
        
        # 사용자에게 즉시 응답
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": f"📰 {first_news['title']}\n\n{summary}\n\n🔗 {first_news['link']}\n\n💬 이 뉴스에 대해 궁금한 점을 물어보세요!"
                        }
                    }
                ],
                "quickReplies": [
                    {
                        "label": "핵심 내용은?",
                        "action": "message",
                        "messageText": "이 뉴스의 핵심은 뭐야?"
                    },
                    {
                        "label": "시장 영향은?",
                        "action": "message",
                        "messageText": "이게 부동산 시장에 어떤 영향을 줄까?"
                    },
                    {
                        "label": "원문 보기",
                        "action": "webLink",
                        "webLinkUrl": first_news['link']
                    }
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"❌ News bot error: {type(e).__name__}: {e}")
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": "뉴스를 처리하는 중 오류가 발생했습니다. 다시 시도해주세요."}}
                ]
            }
        }

async def save_all_news_background(news_items: list, user_id: str):
    """백그라운드에서 모든 뉴스 저장 (필터링 메타데이터 포함)"""
    logger.info(f"🔄 백그라운드 저장 시작: {len(news_items)}개")
    saved_count = 0
    
    for idx, news_item in enumerate(news_items):
        try:
            # Rate Limit 방지
            if idx > 0:
                await asyncio.sleep(2)
            
            # 🆕 키 이름 통일 (link → url)
            if 'link' in news_item and 'url' not in news_item:
                news_item['url'] = news_item['link']
            
            # user_id 추가
            news_item['user_id'] = user_id
            
            # 필터링 메타데이터가 없는 경우 기본값 설정
            if 'is_relevant' not in news_item:
                news_item['is_relevant'] = True
                news_item['relevance_score'] = 50
                news_item['keywords'] = []
                news_item['region'] = ''
                news_item['has_price'] = False
                news_item['has_policy'] = False
                news_item['reason'] = 'Filtering module not available'
            
            # 저장 (필터링 정보 포함)
            save_news_to_csv(news_item)
            save_news_to_gsheet(news_item)
            
            saved_count += 1
            logger.info(
                f"✅ [{saved_count}/{len(news_items)}] 저장 완료 "
                f"[{news_item.get('relevance_score', 0)}점] "
                f"{news_item['title'][:30]}..."
            )
            
        except Exception as e:
            logger.error(f"❌ 뉴스 {idx+1} 저장 실패: {e}")
            logger.error(f"   news_item keys: {news_item.keys()}")
            continue
    
    logger.info(f"🎉 백그라운드 저장 완료: {saved_count}개")

@app.post("/custom")
async def generate_custom(request: RequestBody):
    """REXA 부동산 전문 챗봇 with RAG - 카카오톡 5초 제한 대응"""
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

@app.get("/debug/env")
async def debug_env():
    """환경 변수 체크 (디버깅용)"""
    return {
        "google_sheets_credentials_exists": bool(GOOGLE_SHEETS_CREDENTIALS),
        "google_sheets_credentials_length": len(GOOGLE_SHEETS_CREDENTIALS) if GOOGLE_SHEETS_CREDENTIALS else 0,
        "google_sheets_spreadsheet_id_exists": bool(GOOGLE_SHEETS_SPREADSHEET_ID),
        "google_sheets_spreadsheet_id": GOOGLE_SHEETS_SPREADSHEET_ID if GOOGLE_SHEETS_SPREADSHEET_ID else "NOT_SET",
        "gspread_available": GSPREAD_AVAILABLE,
        "gsheet_client_initialized": gsheet_client is not None,
        "gsheet_worksheet_initialized": gsheet_worksheet is not None,
        "naver_client_id_exists": bool(NAVER_CLIENT_ID),
        "naver_client_secret_exists": bool(NAVER_CLIENT_SECRET)
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
    
    # CSV 초기화
    csv_success = init_csv_file()
    if csv_success:
        logger.info("✅ CSV logging enabled (with filtering columns)")
    else:
        logger.warning("⚠️ CSV logging disabled")
    
    # Google Sheets 초기화
    gsheet_success = init_google_sheets()
    if gsheet_success:
        logger.info("✅ Google Sheets logging enabled (with filtering columns)")
    else:
        logger.warning("⚠️ Google Sheets logging disabled")
    
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
    logger.info(f"   - CSV logging: {'enabled' if csv_success else 'disabled'}")
    logger.info(f"   - Google Sheets: {'enabled' if gsheet_success else 'disabled'}")
    logger.info("="*70)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("👋 Shutting down REXA server (Solar + RAG + News)...")
    await close_redis()
    logger.info("✅ REXA server shut down successfully")
