import sys
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[2] / "src" / "sft_server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
