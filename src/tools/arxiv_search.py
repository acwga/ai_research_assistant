"""
Arxiv 搜索工具
使用 langchain_community 提供的 ArxivQueryRun 工具封装
"""
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool

def get_arxiv_tool():
    """
    创建并返回Arxiv搜索工具。
    """
    arxiv_api_wrapper = ArxivAPIWrapper(
        top_k_results=5,
        ARXIV_MAX_QUERY_LENGTH=300,
        continue_on_failure=True,
        load_max_docs=10
    )

    arxiv__tool = ArxivQueryRun(
        api_wrapper=arxiv_api_wrapper,
        description=(
            "一个用于搜索 arXiv 学术论文的工具。 "
            "输入应该是英文的学术关键词或短语，例如："
            "'attention is all you need', 'LoRA fine-tuning', 'diffusion models survey'。"
            "会返回论文标题、作者、摘要、pdf链接等信息。"
            "适合查找最新论文、技术综述、方法论。"
        )
    )

    return arxiv__tool

if __name__ == "__main__":
    tool = get_arxiv_tool()
    result = tool.invoke("retrieval augmented generation survey")
    print(result)