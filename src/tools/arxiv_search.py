"""
Arxiv 搜索工具
使用 langchain_community 提供的 ArxivQueryRun 工具封装
"""
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool
import concurrent.futures
import time
import threading

# 限流配置：每分钟最多调用次数
RATIO_LIMIT_PER_MINUTE = 10
_arxiv_call_history = []          # 存储调用时间戳
_arxiv_lock = threading.Lock()    # 线程安全锁

arxiv_api_wrapper = ArxivAPIWrapper(
        top_k_results=5,
        ARXIV_MAX_QUERY_LENGTH=300,
        continue_on_failure=True,
        load_max_docs=10
    )

arxiv_tool = ArxivQueryRun(
    api_wrapper=arxiv_api_wrapper,
    description=(
        "一个用于搜索 arXiv 学术论文的工具。 "
        "输入应该是英文的学术关键词或短语，例如："
        "'attention is all you need', 'LoRA fine-tuning', 'diffusion models survey'。"
        "会返回论文标题、作者、摘要、pdf链接等信息。"
        "适合查找最新论文、技术综述、方法论。"
    )
)

@tool(parse_docstring=True)
def search_arxiv(query: str) -> str:
    """
    搜索 arXiv 学术论文

    Args:
        query: 搜索关键词，例如 "retrieval augmented generation"
    
    Returns:
        论文标题、作者、摘要等信息
    """

    # 限流检查
    with _arxiv_lock:
        now = time.time()
        # 清理超过 60 秒的历史记录
        while _arxiv_call_history and _arxiv_call_history[0] < now - 60:
            _arxiv_call_history.pop(0)
        if len(_arxiv_call_history) >= RATIO_LIMIT_PER_MINUTE:
            return f"请求过于频繁，请稍后再试。限制：每分钟最多 {RATIO_LIMIT_PER_MINUTE} 次调用。"
        _arxiv_call_history.append(now)

    # 超时未响应应对机制
    timeout_seconds = 30
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(arxiv_tool.invoke, query)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            return "请求超时，arXiv 服务长时间响应，请稍后再试。"

    return arxiv_tool.invoke(query)

if __name__ == "__main__":
    result = search_arxiv.invoke({"query": "retrieval augmented generation survey"})
    print(result)