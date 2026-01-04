@echo off
echo Starting YouTube AI Chatbot...

echo Starting Backend...
start "Backend API" cmd /k "venv\Scripts\activate && pip install --upgrade yt-dlp >nul 2>&1 && python backend\main.py"

echo Starting Frontend...
start "Frontend App" cmd /k "cd frontend && npm run dev"

echo Services started! 
echo Open http://localhost:3000 to chat.
pause
