import sys
import importlib

required_imports = [
    "dotenv",
    "fastapi",
    "pydantic",
    "langchain_google_genai",
    "langchain_community",
    "langchain_text_splitters",
    "langchain.chains",
    "langchain.memory",
    "faiss",
    "youtube_transcript_api"
]

print("Checking imports...")
missing = []
for module in required_imports:
    try:
        importlib.import_module(module)
        print(f"[OK] {module}")
    except ImportError as e:
        print(f"[FAIL] {module}: {e}")
        missing.append(module)

if missing:
    print(f"\nMISSING MODULES: {missing}")
    sys.exit(1)
else:
    print("\nAll imports successful!")
    sys.exit(0)
