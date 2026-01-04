@echo off
echo Updating dependencies...
call venv\Scripts\activate
pip install --upgrade yt-dlp webvtt-py google-generativeai langchain-google-genai
echo Done!
pause
