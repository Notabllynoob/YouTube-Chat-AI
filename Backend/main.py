import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain



load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


app = FastAPI(
    title="YouTube AI Assistant API",
    description="An API to process YouTube video transcripts and answer questions.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#pydantic
class VideoRequest(BaseModel):
    youtube_url: str
    session_id: str


class QuestionRequest(BaseModel):
    question: str
    session_id: str



user_sessions = {}



def get_transcript_text(youtube_url: str):
    try:
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        transcript = loader.load()
        return "".join([doc.page_content for doc in transcript])
    except Exception as e:
        print(f"[Transcript Error] {youtube_url} — {type(e).__name__}: {e}")
        return None




def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def create_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")  # Log the error
        return None


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)




@app.post("/process-video")
async def process_video(request: VideoRequest):
    raw_text = get_transcript_text(request.youtube_url)
    if not raw_text:
        raise HTTPException(status_code=400, detail="Failed to retrieve or process transcript.")

    text_chunks = get_text_chunks(raw_text)
    vector_store = create_vector_store(text_chunks)

    if not vector_store:
        raise HTTPException(status_code=500, detail="Failed to create vector store.")

    user_sessions[request.session_id] = vector_store
    return {"status": "success", "message": "Video processed. You can now ask questions."}


@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    if request.session_id not in user_sessions:
        raise HTTPException(status_code=404, detail="Invalid session. Please process a video first.")

    vector_store = user_sessions[request.session_id]
    docs = vector_store.similarity_search(request.question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": request.question}, return_only_outputs=True)

    return {"answer": response["output_text"]}