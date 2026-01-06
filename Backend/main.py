import os
import uuid
import logging
import sqlite3
import json
import yt_dlp
import webvtt
import glob
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.critical("GOOGLE_API_KEY not found in environment variables.")

app = FastAPI(title="YouTube AI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

TEMP_PATH = "./temp_subs"
os.makedirs(TEMP_PATH, exist_ok=True)

db_conn = sqlite3.connect(":memory:", check_same_thread=False)

def init_db():
    c = db_conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (session_id TEXT PRIMARY KEY, video_title TEXT, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, timestamp TEXT)''')
    db_conn.commit()

init_db()

from fastapi import Header, Request

def get_api_key(x_gemini_api_key: str | None = Header(default=None)):
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key
    
    if x_gemini_api_key:
        return x_gemini_api_key
        
    raise HTTPException(status_code=401, detail="Missing API Key. Please provide a Google API Key in settings.")

class VideoRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    question: str
    session_id: str

class YtDlpLoader:
    def __init__(self, url):
        self.url = url

    def load(self):
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-orig', 'en-GB'],
            'outtmpl': f'{TEMP_PATH}/%(id)s',
            'quiet': True,
            'no_warnings': True,
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }

        # Handle Cookies: Priority 1: Render Secret File, Priority 2: Env Var
        cookies_path = None
        render_secret_path = "/etc/secrets/cookies.txt"
        
        if os.path.exists(render_secret_path):
            logger.info(f"Found Render Secret File at {render_secret_path}")
            cookies_path = render_secret_path
            ydl_opts['cookiefile'] = cookies_path
        else:
            # Fallback to Env Var
            cookies_content = os.getenv("YOUTUBE_COOKIES")
            if cookies_content:
                logger.info("Found YOUTUBE_COOKIES env var. Creating temp cookie file.")
                cookies_path = f"{TEMP_PATH}/cookies_{uuid.uuid4()}.txt"
                with open(cookies_path, "w") as f:
                    f.write(cookies_content)
                ydl_opts['cookiefile'] = cookies_path
            else:
                logger.warning("No cookies found (neither /etc/secrets/cookies.txt nor YOUTUBE_COOKIES). Bot detection likely.")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
                if not info:
                    raise Exception("Failed to extract video info (bot detection or empty response)")
                
                video_title = info.get('title', 'Unknown Title')
                video_id = info.get('id')

            vtt_files = glob.glob(f"{TEMP_PATH}/{video_id}*.vtt")
            
            if not vtt_files:
                 return [], video_title

            vtt_file = vtt_files[0]
            text_content = ""
            
            for caption in webvtt.read(vtt_file):
                text_content += caption.text + " "
            
            try:
                os.remove(vtt_file)
            except:
                pass

            return [Document(page_content=text_content.strip(), metadata={"title": video_title, "source": self.url})], video_title

        except Exception as e:
            logger.error(f"yt-dlp error: {e}")
            raise e
        finally:
            # Cleanup cookies file
            if cookies_path and os.path.exists(cookies_path):
                try:
                    os.remove(cookies_path)
                except:
                    pass

def get_chat_history(session_id):
    c = db_conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    return [(r[0], r[1]) for r in rows]

def add_message(session_id, role, content):
    c = db_conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
              (session_id, role, content, datetime.now().isoformat()))
    db_conn.commit()

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "YouTube AI Chatbot API is running"}

import chromadb
global_chroma_client = chromadb.Client()

@app.post("/api/process")
async def process_video(request: VideoRequest, x_gemini_api_key: str | None = Header(default=None)):
    api_key = get_api_key(x_gemini_api_key)
         
    try:
        session_id = str(uuid.uuid4())
        logger.info(f"Processing URL: {request.url} for Session: {session_id}")

        loader = YtDlpLoader(request.url)
        docs, video_title = loader.load()
        
        if not docs or not docs[0].page_content:
            raise HTTPException(status_code=400, detail="Could not retrieve transcript. Video might not have English captions.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        if not splits:
             raise HTTPException(status_code=400, detail="Transcript was empty.")

        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
        
        Chroma.from_documents(
            client=global_chroma_client,
            documents=splits, 
            embedding=embeddings, 
            collection_name=f"session_{session_id}"
        )
        
        c = db_conn.cursor()
        c.execute("INSERT INTO sessions (session_id, video_title, created_at) VALUES (?, ?, ?)",
                  (session_id, video_title, datetime.now().isoformat()))
        db_conn.commit()
        
        return {
            "status": "success", 
            "message": "Video processed!",
            "session_id": session_id,
            "video_title": video_title
        }
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        msg = str(e)
        if "HTTP Error 429" in msg:
            msg = "Rate limit reached. If you are using a clear key, try setting your own API Key."
        raise HTTPException(status_code=500, detail=msg)

@app.post("/api/chat")
async def chat(request: ChatRequest, x_gemini_api_key: str | None = Header(default=None)):
    api_key = get_api_key(x_gemini_api_key)

    session_id = request.session_id
    
    c = db_conn.cursor()
    c.execute("SELECT video_title FROM sessions WHERE session_id = ?", (session_id,))
    session = c.fetchone()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please re-process the video.")
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
        
        vector_store = Chroma(
            client=global_chroma_client,
            collection_name=f"session_{session_id}",
            embedding_function=embeddings,
        )

        raw_history = get_chat_history(session_id)
        chat_history_tuples = []
        temp_q = None
        for role, content in raw_history:
            if role == 'user':
                temp_q = content
            elif role == 'assistant' and temp_q:
                chat_history_tuples.append((temp_q, content))
                temp_q = None

        llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.7, google_api_key=api_key)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        from langchain.prompts import PromptTemplate

        custom_template = """You are an AI assistant analyzing a YouTube video transcript. Use the following pieces of transcript context to answer the question about the video. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
Helpful Answer (refer to "the video" or the content, not "the text"):"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(custom_template)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        result = qa_chain.invoke({
            "question": request.question, 
            "chat_history": chat_history_tuples
        })
        
        answer = result['answer']
        
        add_message(session_id, "user", request.question)
        add_message(session_id, "assistant", answer)
            
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
