# YouTube AI Chatbot (RAG) 🎥🤖

**Chat with any YouTube video.** This AI Assistant processes YouTube video transcripts and lets you ask questions about the content in real-time.

🚀 **Live Demo:** [https://youtube-chat-ai.vercel.app/](https://youtube-chat-ai.vercel.app/)

---

## 🛠️ Tech Stack

This project is built using a modern **RAG (Retrieval-Augmented Generation)** architecture:

*   **Frontend**: Next.js (React), TailwindCSS, Lucide Icons
*   **Backend**: FastAPI (Python), Dockerized
*   **LLM Integration**: LangChain, Google Gemini Pro 1.5
*   **Vector Database**: ChromaDB (Local persistent storage)
*   **Video Processing**: `yt-dlp` (Subtitle extraction)

---

## 📖 How to Use

1.  **Open the App**: Go to the [Live Demo](https://youtube-chat-ai.vercel.app/).
2.  **Add API Key**:
    *   Click the **Settings (Gear Icon)** in the top right.
    *   Enter your **Google Gemini API Key**. (You can get one for free from Google AI Studio).
    *   *Note: Your key is stored locally in your browser session.*
3.  **Choose a Video**:
    *   Paste a YouTube link into the input box.
    *   **Pro Tip**: Works best with videos that have **existing subtitles/captions**.
    *   **Length Recommendation**: 10-20 minutes is ideal for the best speed and accuracy.
4.  **Chat**: Ask anything! "What is the summary?", "What did he say about X?", etc.

---

## 🏗️ Architecture

1.  **Ingestion**: The backend downloads the subtitles using `yt-dlp`.
2.  **Chunking**: Text is split into chunks of 1000 characters.
3.  **Embedding**: Chunks are converted into vector embeddings using Google's embedding model.
4.  **Retrieval**: When you ask a question, the system finds the top 4 most relevant chunks from ChromaDB.
5.  **Generation**: The relevant context + your question is sent to Gemini to generate the answer.

---

## 🚀 Local Development

1.  **Clone the repo**
    ```bash
    git clone https://github.com/Notabllynoob/YouTube-Chat-AI.git
    cd YouTube-Chat-AI
    ```

2.  **Backend** (Docker)
    ```bash
    cd Backend
    # Create .env with GOOGLE_API_KEY
    docker-compose up --build
    ```

3.  **Frontend**
    ```bash
    cd Frontend
    npm install
    npm run dev
    ```
