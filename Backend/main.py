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

# --- Persistence Setup ---
DB_PATH = "./db/chat_history.db"
CHROMA_PATH = "./db/chroma_db"
TEMP_PATH = "./temp_subs"

# Ensure directories exist
os.makedirs("./db", exist_ok=True)
os.makedirs(TEMP_PATH, exist_ok=True)

# Initialize SQLite
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (session_id TEXT PRIMARY KEY, video_title TEXT, created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

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
            # Prefer English, then others. 
            # yt-dlp will auto-translate if we ask, but 'all' is safer to just get whatever exists.
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

            # yt-dlp saves as [id].en.vtt or similar. Find the file.
            vtt_files = glob.glob(f"{TEMP_PATH}/{video_id}*.vtt")
            
            if not vtt_files:
                 # Fallback: Try to get ANY subtitle if English failed
                 return [], video_title

            # Parse the first VTT file found
            vtt_file = vtt_files[0]
            text_content = ""
            
            for caption in webvtt.read(vtt_file):
                text_content += caption.text + " "
            
            # Cleanup temp file
            try:
                os.remove(vtt_file)
            except:
                pass

            return [Document(page_content=text_content.strip(), metadata={"title": video_title, "source": self.url})], video_title

        except Exception as e:
            logger.error(f"yt-dlp error: {e}")
            raise e

def get_chat_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]

def add_message(session_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
              (session_id, role, content, datetime.now().isoformat()))
    conn.commit()
    conn.close()

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "YouTube AI Chatbot API (Persistent + yt-dlp) is running"}

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

        # 4. Create/Update Persistent Vector Store
        # We use a unique collection name per session to isolate different videos
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
        
        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=CHROMA_PATH, 
            collection_name=f"session_{session_id}"
        )
        
        # 5. Store Session Metadata
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO sessions (session_id, video_title, created_at) VALUES (?, ?, ?)",
                  (session_id, video_title, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        return {
            "status": "success", 
            "message": "Video processed!",
            "session_id": session_id,
            "video_title": video_title
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        # Clean error message
        msg = str(e)
        if "HTTP Error 429" in msg:
            msg = "YouTube is rate-limiting requests (429). Please try again later."
        raise HTTPException(status_code=500, detail=msg)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    
    # Check if session exists in SQL
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT video_title FROM sessions WHERE session_id = ?", (session_id,))
    session = c.fetchone()
    conn.close()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please re-process the video.")
    
    try:
        # Load Existing Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
        vector_store = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings,
            collection_name=f"session_{session_id}"
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

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=GOOGLE_API_KEY)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            verbose=True
        )
        
        # Invoke Chain
        result = qa_chain.invoke({
            "question": request.question, 
            "chat_history": chat_history_tuples
        })
        
        answer = result['answer']
        
        # Save new messages to DB
        add_message(session_id, "user", request.question)
        add_message(session_id, "assistant", answer)
            
        return {"answer": answer}
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
