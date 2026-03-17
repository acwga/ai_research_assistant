import os
from dotenv import load_dotenv
load_dotenv()
print("GITHUB_TOKEN 是否存在：", bool(os.getenv("GITHUB_TOKEN")))
print("LANGCHAIN_TRACING_V2 是否存在：", bool(os.getenv("LANGCHAIN_TRACING_V2")))
print("LANGCHAIN_API_KEY 是否存在：", bool(os.getenv("LANGCHAIN_API_KEY")))
print("LANGCHAIN_PROJECT 是否存在：", bool(os.getenv("LANGCHAIN_PROJECT")))