import os
from dotenv import load_dotenv

load_dotenv()

# ==============================
# LLM 配置
# ==============================
MODEL_NAME = "qwen-max"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TEMPERATURE = 0.7
MAX_TOKENS = 4096

# ==============================
# 其他全局常量
# ==============================
MAX_SEARCH_RESULTS = 5
DEFAULT_LANGUAGE = "python"
CHROMA_COLLECTION_NAME = "research_memory"

# ==============================
# API Keys
# ==============================
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")