# YouTube Chat AI 🎥🤖

A powerful, private, and modern AI chatbot that watches YouTube videos for you. Paste a link, and chat with the video content instantly.

![YouTube Chat AI UI](https://github.com/Notabllynoob/YouTube-Chat-AI/assets/placeholder/ui_screenshot.png)
*(Note: Replace with actual screenshot link if available)*

## Features

*   **📺 Video Analysis**: Summarize, question, and explore any YouTube video.
*   **🧠 Advanced AI**: Powered by Google's **Gemini 2.5 Flash** for fast, accurate responses.
*   **🔒 Privacy First**: Ephemeral session handling. All data is wiped when you close the tab or restart the server.
*   **💅 Modern UI**: Clean, full-screen "YouTube-themed" interface with glassmorphism and dark mode.
*   **🏃‍♂️ Robust Backend**: Uses `yt-dlp` to reliably fetch transcripts, avoiding common IP blocks.

## 🚀 Getting Started

You will need a **Google Gemini API Key**. [Get it here](https://aistudio.google.com/app/apikey).

### Option 1: Docker (Recommended)

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Notabllynoob/YouTube-Chat-AI.git
    cd YouTube-Chat-AI
    ```

2.  **Configure Environment**:
    *   Navigate to the `Backend` folder.
    *   Rename `.env.example` to `.env`.
    *   Open `.env` and paste your `GOOGLE_API_KEY`.
    
    *(Alternatively, you can pass the key directly in `docker-compose.yml` or your deployment environment variables).*

3.  **Run with Docker Compose**:
    ```bash
    docker-compose up --build
    ```

4.  **Open the App**:
    Visit [http://localhost:3000](http://localhost:3000).

---

### Option 2: Run Locally (Dev)

**Backend:**
```bash
cd Backend
# Create virtual env (optional but recommended)
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup .env (as above)
# Run Server
python main.py
```

**Frontend:**
```bash
cd Frontend
npm install
npm run dev
```

## 🛠️ Tech Stack

*   **Frontend**: Next.js, Tailwind CSS, TypeScript
*   **Backend**: FastAPI, LangChain, ChromaDB (In-Memory), SQLite (In-Memory)
*   **AI Models**: Gemini 2.5 Flash (Chat), Text-Embedding-004

## License

MIT
