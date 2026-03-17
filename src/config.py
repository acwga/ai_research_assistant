import os
from dotenv import load_dotenv

load_dotenv()

# ==============================
# LLM 配置
# ==============================
LLM_PROVIDER = "dashscope"

# 根据 provider 选择不同的模型
LLM_CONFIG = {
    "dashscope": {
        "model_name": "qwen-max",
        "temperature": 0.7,
        "max_tokens": 4096
    },
    "ollama": {
        "model_name": "deepseek-r1:7b",
        "base_url": "http://localhost:11434/v1",
        "temperature": 0.7
    }
}

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

def get_llm_config():
    """根据当前 LLM_PROVIDER 获取对应的配置"""
    if LLM_PROVIDER not in LLM_CONFIG:
        raise ValueError(f"不支持的 LLM provider: {LLM_PROVIDER}")
    return LLM_CONFIG[LLM_PROVIDER]

if __name__ == "__main__":
    print("当前 LLM Provider:", LLM_PROVIDER)
    print("LLM 配置:", get_llm_config())
    print("GITHUB_TOKEN 是否存在:", bool(GITHUB_TOKEN))
    print("DASHSCOPE_API_KEY 是否存在:", bool(DASHSCOPE_API_KEY))