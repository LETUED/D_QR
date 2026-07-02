from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ORCHESTRATOR_MODEL = "claude-sonnet-4-6"
AGENT_MODEL = "claude-sonnet-4-6"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
