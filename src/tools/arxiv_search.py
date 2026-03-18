"""
Arxiv 搜索工具
使用 langchain_community 提供的 ArxivQueryRun 工具封装
"""
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool

@tool
def search_arxiv(query: str) -> str:
    """
    搜索 arXiv 学术论文

    Args:
        query: 搜索关键词，例如 "retrieval augmented generation"
    
    Returns:
        论文标题、作者、摘要等信息
    """
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

    return arxiv_tool.invoke(query)

if __name__ == "__main__":
    result = search_arxiv.invoke({"query": "retrieval augmented generation survey"})
    print(result)