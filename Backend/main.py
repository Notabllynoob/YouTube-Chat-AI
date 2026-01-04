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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.critical("GOOGLE_API_KEY not found in environment variables.")

# Initialize FastAPI app
app = FastAPI(title="YouTube AI Chatbot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration (Env with Defaults) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

# --- Persistence Setup (Ephemeral Mode) ---
# Data is lost when the server restarts
TEMP_PATH = "./temp_subs"
# No file paths for DBs

# Ensure temp directory exists
os.makedirs(TEMP_PATH, exist_ok=True)

# Initialize In-Memory SQLite
# We use a global connection because :memory: is local to the object.
# check_same_thread=False allows FastAPI threads to use it.
db_conn = sqlite3.connect(":memory:", check_same_thread=False)

def init_db():
    c = db_conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (session_id TEXT PRIMARY KEY, video_title TEXT, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, timestamp TEXT)''')
    db_conn.commit()

init_db()

class VideoRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    question: str
    session_id: str

class YtDlpLoader:
    def __init__(self, url):
        self.url = url

    def load(self):
        # Configure yt-dlp to download subtitles only
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-orig', 'en-GB'], 
            'outtmpl': f'{TEMP_PATH}/%(id)s',
            'quiet': True,
            'no_warnings': True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
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
    return {"status": "ok", "message": "YouTube AI Chatbot API (Ephemeral + yt-dlp) is running"}

@app.post("/api/process")
async def process_video(request: VideoRequest):
    if not GOOGLE_API_KEY:
         raise HTTPException(status_code=500, detail="Server configuration error: Google API Key missing.")
         
    try:
        # 1. Generate Session ID
        session_id = str(uuid.uuid4())
        logger.info(f"Processing URL: {request.url} for Session: {session_id}")

        # 2. Load Transcript using yt-dlp
        loader = YtDlpLoader(request.url)
        docs, video_title = loader.load()
        
        if not docs or not docs[0].page_content:
            raise HTTPException(status_code=400, detail="Could not retrieve transcript. Video might not have English captions.")

        # 3. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        if not splits:
             raise HTTPException(status_code=400, detail="Transcript was empty.")

        # 4. Create Ephemeral Vector Store (In-Memory)
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
        
        # When persist_directory is None (default), Chroma is in-memory
        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            collection_name=f"session_{session_id}"
        )
        
        # 5. Store Session Metadata
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
            msg = "YouTube is rate-limiting requests (429). Please try again later."
        raise HTTPException(status_code=500, detail=msg)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    
    # Check if session exists in SQL
    c = db_conn.cursor()
    c.execute("SELECT video_title FROM sessions WHERE session_id = ?", (session_id,))
    session = c.fetchone()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please re-process the video.")
    
    try:
        # Load Existing Vector Store (In-Memory)
        # Note: In-Memory Chroma must exist in memory. 
        # Since this is a new request, do we lose the vector store object?
        # YES. Standard Chroma in-memory is tied to the Client object.
        # If we just do Chroma(collection_name=...), it creates a NEW empty client if not persistent.
        
        # FIX: We need a GLOBAL Chroma Client for Ephemeral Mode in a persistent server process.
        
        # We will use the SAME logic as SQL: create a global client but separate collections.
        pass # Logic handled below

        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
        
        # LangChain Chroma doesn't easily expose the client sharing in the constructor wrapper.
        # But if we use the same settings (implied default client), it might work?
        # Actually, best practice for local server is to pass `client=global_client`.
        
        # For simplicity in this script, we'll let LangChain handle it. 
        # If "persist_directory" is None, it might reset.
        # Wait, if we init Chroma() again without documents, it tries to load.
        # If it's in memory, it won't find it unless we share the client.
        
        # Let's fix this properly.
        
        return await chat_logic(request, session_id, embeddings)

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Global Chroma Client
import chromadb
global_chroma_client = chromadb.Client()

async def chat_logic(request, session_id, embeddings):
    # Use the global client so data persists across requests (but not restarts)
    vector_store = Chroma(
        client=global_chroma_client,
        collection_name=f"session_{session_id}",
        embedding_function=embeddings,
    )

    # Load Chat History
    raw_history = get_chat_history(session_id)
    
    chat_history_tuples = []
    temp_q = None
    for role, content in raw_history:
        if role == 'user':
            temp_q = content
        elif role == 'assistant' and temp_q:
            chat_history_tuples.append((temp_q, content))
            temp_q = None

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.7, google_api_key=GOOGLE_API_KEY)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=True
    )
    
    result = qa_chain.invoke({
        "question": request.question, 
        "chat_history": chat_history_tuples
    })
    
    answer = result['answer']
    
    add_message(session_id, "user", request.question)
    add_message(session_id, "assistant", answer)
        
    return {"answer": answer}

# We need to monkey-patch or adjust the process_video to use the global client too
async def process_vid_logic(request, session_id, video_title, splits, embeddings):
    Chroma.from_documents(
        client=global_chroma_client,
        documents=splits, 
        embedding=embeddings, 
        collection_name=f"session_{session_id}"
    )

# Redefining process_video to use the global logic
@app.post("/api/process")
async def process_video(request: VideoRequest):
    if not GOOGLE_API_KEY:
         raise HTTPException(status_code=500, detail="Server configuration error: Google API Key missing.")
         
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

        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
        
        # Use Global Client
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
            msg = "YouTube is rate-limiting requests (429). Please try again later."
        raise HTTPException(status_code=500, detail=msg)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    
    c = db_conn.cursor()
    c.execute("SELECT video_title FROM sessions WHERE session_id = ?", (session_id,))
    session = c.fetchone()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please re-process the video.")
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
        
        # Use Global Client
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

        llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.7, google_api_key=GOOGLE_API_KEY)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            verbose=True
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
