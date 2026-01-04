# Maintenance Guide 🛠️

This project is designed to be low-maintenance, but YouTube changes its site frequently, and AI models evolve.

## 1. Auto-Updates
The `start_app.bat` script now automatically updates `yt-dlp` (the YouTube downloader) every time you run it. This prevents "IP Blocking" errors caused by outdated extractors.

## 2. Changing AI Models
If Google releases a new model (e.g., `gemini-3.0`) or changes names, you DON'T need to edit code.
1.  Open `Backend/.env`
2.  Add/Edit these lines:
    ```env
    CHAT_MODEL=gemini-2.5-flash
    EMBEDDING_MODEL=models/text-embedding-004
    ```

## 3. Full Update using `update.bat`
If the app stops working or you want the latest AI library features:
1.  Run `update.bat`.
2.  This upgrades `yt-dlp`, `google-generativeai`, and `langchain` to their latest versions.

## 4. Resetting Storage
If the database gets corrupted or you want to clear all history:
1.  Delete the `db` folder.
2.  The app will recreate it automatically on next run.

## 5. Troubleshooting "Port in Use"
If you see "Port 3000 is in use" or "Unable to acquire lock":
1.  This means the app is already running in the background.
2.  Run this command in terminal to kill it:
    ```cmd
    taskkill /F /IM node.exe
    taskkill /F /IM python.exe
    ```
3.  Then try `start_app.bat` again.
