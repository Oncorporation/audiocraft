# modules/constants.py
# constants.py contains all the constants used in the project
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path)

IS_SHARED_SPACE = "Surn/UnlimitedMusicGen" in os.environ.get('SPACE_ID', '')

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("HF_TOKEN is not set. Please check your .env file.")
try:
    if os.environ['TMPDIR']:
        TMPDIR = os.environ['TMPDIR']
    else:
        TMPDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
except:
    TMPDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')

os.makedirs(TMPDIR, exist_ok=True)

model_extensions = {".glb", ".gltf", ".obj", ".ply"}
model_extensions_list = list(model_extensions)
image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
image_extensions_list = list(image_extensions)
audio_extensions = {".mp3", ".wav", ".ogg", ".flac", ".aac"}
audio_extensions_list = list(audio_extensions)
video_extensions = {".mp4"}
video_extensions_list = list(video_extensions)
upload_file_types = model_extensions_list + image_extensions_list + audio_extensions_list + video_extensions_list

# Constants for URL shortener
HF_REPO_ID = os.getenv("HF_REPO_ID")
if not HF_REPO_ID:
    HF_REPO_ID = "Surn/Storage"  # Replace with your Hugging Face repository ID
SHORTENER_JSON_FILE = "shortener.json"  # The name of your JSON file in the repo
