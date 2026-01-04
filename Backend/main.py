import os
import uuid
import logging
import sqlite3
import json
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
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

# Ensure db directory exists
os.makedirs("./db", exist_ok=True)

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
    return {"status": "ok", "message": "YouTube AI Chatbot API (Persistent) is running"}

@app.post("/api/process")
async def process_video(request: VideoRequest):
    if not GOOGLE_API_KEY:
         raise HTTPException(status_code=500, detail="Server configuration error: Google API Key missing.")
         
    try:
        # 1. Generate Session ID
        session_id = str(uuid.uuid4())
        logger.info(f"Processing URL: {request.url} for Session: {session_id}")

        # 2. Load Transcript
        loader = YoutubeLoader.from_youtube_url(
            request.url, 
            add_video_info=True,
            language=["en", "en-US"],
            translation="en"
        )
        docs = loader.load()
        
        if not docs:
            raise HTTPException(status_code=400, detail="Could not retrieve transcript.")

        # 3. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        if not splits:
             raise HTTPException(status_code=400, detail="Transcript was empty.")

        # 4. Create/Update Persistent Vector Store
        # We use a unique collection name per session to isolate different videos
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        
        vector_store = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=CHROMA_PATH,
            collection_name=f"session_{session_id}"
        )
        
        # 5. Store Session Metadata
        video_title = docs[0].metadata.get("title", "Unknown Title")
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
        raise HTTPException(status_code=500, detail=str(e))

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
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings,
            collection_name=f"session_{session_id}"
        )
        
        # Load Chat History
        raw_history = get_chat_history(session_id)
        # LangChain expects list of (question, answer) tuples for history
        # Our SQL stores individual messages. We need to pair them up or just pass them as context.
        # For ConversationalRetrievalChain, we typically pass [(q, a), (q, a)]
        
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
