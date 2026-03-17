"""
论文分析工具
从论文摘要中提取关键信息，用于后续报告生成
"""
from langchain_core.tools import tool
from src.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME
from openai import OpenAI

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL
)

@tool
def analyze_paper(paper_title: str, abstract: str) -> str:
    """
    分析论文摘要，提取核心贡献、方法、实验结果和优缺点
    
    Args:
        paper_title: 论文标题
        abstract: 论文摘要内容
    
    Returns:
        结构化分析结果
    """
    prompt = f"""
    请分析以下论文摘要，提取关键信息：
    
    论文标题：{paper_title}
    摘要内容：{abstract}
    
    请按以下格式输出：
    
    ## 核心贡献
    [一句话总结论文的核心贡献]
    
    ## 主要方法
    [列出1-3个主要方法或技术]
    
    ## 实验结果
    [如果有实验结果，简要总结]
    
    ## 优缺点
    - 优点：[列出2-3个优点]
    - 局限：[列出1-2个局限性]
    
    ## 适用场景
    [这个技术最适合用在什么场景]
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的论文分析助手，能够快速提取论文核心信息。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        analysis = response.choices[0].message.content

        # 添加论文引用信息
        result = f"## 📄 {paper_title}\n\n{analysis}\n\n---"
        return result
    
    except Exception as e:
        return f"论文分析失败：{str(e)}"
    
@tool
def compare_papers(papers_info: list) -> str:
    """
    对比多篇论文的核心方法
    
    Args:
        papers_info: 包含论文标题和摘要的列表，格式为 [{"title": "论文1", "abstract": "摘要1"}, ...]
    
    Returns:
        论文对比分析结果
    """
    if not papers_info or len(papers_info) < 2:
        return "需要至少2篇论文进行对比"
    
    # 构建对比提示
    papers_text = ""
    for i, paper in enumerate(papers_info, 1):
        papers_text += f"论文{i}：{paper['title']}\n摘要：{paper['abstract'][:300]}...\n\n"

    prompt = f"""
    请对比以下{len(papers_info)}篇论文：
    
    {papers_text}
    
    请按以下格式输出对比分析：
    
    ## 方法对比
    [各论文使用的主要方法对比]
    
    ## 创新点对比
    [各论文的创新之处对比]
    
    ## 优缺点对比
    [各论文的优缺点对比]
    
    ## 总结建议
    [根据对比结果，给出研究建议]
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的论文对比分析专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        return response.choices[0].message.content
    
    except Exception as e:
        return f"论文对比失败：{str(e)}"
    
if __name__ == "__main__":
    test_title = "Attention Is All You Need"
    test_abstract = "The dominant sequence transduction models " \
    "are based on complex recurrent or convolutional neural " \
    "networks that include an encoder and a decoder. " \
    "The best performing models also connect the encoder " \
    "and decoder through an attention mechanism. We propose a " \
    "new simple network architecture, the Transformer, " \
    "based solely on attention mechanisms, dispensing with " \
    "recurrence and convolutions entirely. Experiments on two " \
    "machine translation tasks show these models to be superior " \
    "in quality while being more parallelizable and requiring " \
    "significantly less time to train."

    result = analyze_paper.invoke({
        "paper_title": test_title,
        "abstract": test_abstract
    })
    print(result)