"""
报告生成工具
整合搜索结果和分析，生成结构化研究报告
"""
from langchain_core.tools import tool
from src.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME
from openai import OpenAI
from datetime import datetime
from src.prompts import (
    REPORT_WRITING_SYSTEM_PROMPT,
    REPORT_WRITING_USER_PROMPT_TEMPLATE,
    FINDINGS_SUMMARY_SYSTEM_PROMPT,
    FINDINGS_SUMMARY_USER_PROMPT_TEMPLATE
)

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL
)

@tool(parse_docstring=True)
def write_report(topic: str, research_data: dict) -> str:
    """
    根据研究数据生成最终报告
    
    Args:
        topic: 研究主题
        research_data: 包含论文、GitHub仓库、代码示例、对比分析等信息的字典， 结构如下: {"papers": [...], "github": [...], "code_examples": [...], "comparisons": [...]}
    
    Returns:
        结构化的研究报告文本
    """
    # 整理研究数据
    papers_section = ""
    if research_data.get("papers"):
        papers_section = "\n".join(research_data["papers"])
    else:
        papers_section = "未找到相关论文。"
    
    github_section = ""
    if research_data.get("github"):
        github_section = "\n".join(research_data["github"])
    else:
        github_section = "未找到相关GitHub仓库。"

    code_section = ""
    if research_data.get("code_examples"):
        code_section = "\n".join(research_data["code_examples"])
    else:
        code_section = "未生成代码示例。"
    
    comparison_section = ""
    if research_data.get("comparisons"):
        comparison_section = "\n".join(research_data["comparisons"])
    else:
        comparison_section = "未进行对比分析。"

    # 生成报告  
    prompt = REPORT_WRITING_USER_PROMPT_TEMPLATE.format(
        topic=topic,
        papers_section=papers_section,
        github_section=github_section,
        code_section=code_section,
        comparison_section=comparison_section
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": REPORT_WRITING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=2000
        )

        report_content = response.choices[0].message.content

        final_report = f"""# {topic} 研究报告

{report_content}

---
生成日期：{datetime.now().strftime("%Y-%m-%d")}
"""
        return final_report
    
    except Exception as e:
        return f"报告生成失败：{str(e)}"
    
@tool
def summarize_findings(topic: str, findings: list) -> str:
    """
    对研究发现进行简短总结
    
    Args:
        topic: 研究主题
        findings: 研究发现的关键点列表
    
    Returns:
        简洁的研究总结
    """
    findings_text = "\n".join([f"- {f}" for f in findings])

    prompt = FINDINGS_SUMMARY_USER_PROMPT_TEMPLATE.format(
        topic=topic,
        findings_text=findings_text
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": FINDINGS_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )

        return response.choices[0].message.content
    
    except Exception as e:
        return f"总结失败：{str(e)}"
    
if __name__ == "__main__":
    print("="*50)
    print("测试：生成研究报告")
    print("="*50)
    
    test_data = {
        "papers": [
            "## 📄 Attention Is All You Need\n\n核心贡献：提出了完全基于注意力机制的Transformer架构。\n主要方法：多头注意力、位置编码。",
            "## 📄 BERT: Pre-training of Deep Bidirectional Transformers\n\n核心贡献：提出双向Transformer预训练方法。\n主要方法：Masked LM、Next Sentence Prediction。"
        ],
        "github": [
            "[huggingface/transformers]\n⭐ 150k  |  🍴 35k\n提供数千种预训练模型的Transformers库\nhttps://github.com/huggingface/transformers"
        ],
        "code_examples": [
            "```python\nfrom transformers import AutoTokenizer, AutoModel\nmodel = AutoModel.from_pretrained('bert-base-uncased')\n```"
        ],
        "comparisons": [
            "## 方法对比\nTransformer：并行计算，适合长距离依赖\nBERT：双向上下文，适合理解任务"
        ]
    }
    
    result = write_report.invoke({
        "topic": "Transformer与BERT在NLP中的应用",
        "research_data": test_data
    })
    print(result)