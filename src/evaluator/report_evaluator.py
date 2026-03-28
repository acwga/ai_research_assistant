"""
报告质量评估器
使用规则和本地 LLM 对研究报告进行多维度打分
"""
import re
from typing import Dict, List, Tuple
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from src.logger import get_logger
from src.prompts import REPORT_EVALUATION_SYSTEM_PROMPT

logger = get_logger("report_evaluator")

class ReportEvaluator:
    """研究报告质量评估器"""

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        """
        初始化评估器，使用轻量本地模型进行语义评估
        """
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0.1)
        # 定义必需章节标题关键词（支持中英文）
        self.required_sections = [
            "研究背景", "背景", "问题定义", "问题描述",
            "相关文献", "文献综述", "技术综述", "相关工作",
            "核心方法", "方法对比", "模型对比",
            "代码实现", "代码示例",
            "优缺点", "适用场景",
            "发展趋势", "未来方向",
            "参考文献", "参考"
        ]
        # 权重：规则总分占 60%，LLM 评分占 40%
        self.rule_weight = 0.6
        self.llm_weight = 0.4

    def evaluate(self, report_text: str) -> Dict:
        """
        对报告进行质量评估

        Args:
            report_text: 完整的 Markdown 报告文本

        Returns:
            包含各项评分和总分的字典
        """
        logger.info("开始评估报告质量")
        # 1. 规则评分
        rule_scores = self._rule_based_score(report_text)
        # 2. LLM 评分
        llm_score, llm_comment = self._llm_based_score(report_text)

        # 综合得分
        total_score = (rule_scores["total"] * self.rule_weight +
                       llm_score * self.llm_weight)

        result = {
            "total_score": round(total_score, 2),
            "rule_based": rule_scores,
            "llm_based": {
                "score": llm_score,
                "comment": llm_comment
            },
            "details": {
                "sections_found": rule_scores.get("sections_found", []),
                "code_blocks": rule_scores.get("code_blocks", 0),
                "has_references": rule_scores.get("has_references", False)
            }
        }
        logger.info(f"评估完成，总分: {result['total_score']}")
        return result

    def _rule_based_score(self, text: str) -> Dict:
        """
        基于规则的评分（满分 10 分）
        """
        score = 0.0
        sections_found = []
        # 1. 章节覆盖度（满分 3 分）
        for sec in self.required_sections:
            pattern = rf"^#+\s*.*{re.escape(sec)}.*$"  # 匹配标题行
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                sections_found.append(sec)
        coverage_score = min(len(sections_found) / len(self.required_sections), 1.0) * 3
        score += coverage_score

        # 2. 代码块数量（满分 2 分）
        code_blocks = len(re.findall(r"```[\s\S]*?```", text))
        code_score = min(code_blocks / 2, 1.0) * 2  # 至少2个代码块满分
        score += code_score

        # 3. 参考文献（满分 2 分）
        has_ref = bool(re.search(r"参考文献|参考", text, re.IGNORECASE))
        ref_score = 2 if has_ref else 0
        score += ref_score

        # 4. 报告长度（满分 2 分）
        char_len = len(text)
        if char_len > 2500:
            len_score = 2
        elif char_len > 1500:
            len_score = 1
        else:
            len_score = 0
        score += len_score

        # 5. 格式整洁性（满分 1 分）
        # 简单检查是否有明显乱码或空行过少
        lines = text.splitlines()
        if all(len(line.strip()) < 200 for line in lines if line.strip()):
            format_score = 1
        else:
            format_score = 0
        score += format_score

        return {
            "total": round(score, 2),
            "sections_found": sections_found,
            "code_blocks": code_blocks,
            "has_references": has_ref,
            "length_chars": char_len
        }

    def _llm_based_score(self, text: str) -> Tuple[float, str]:
        """
        使用本地 LLM 对报告进行评价，返回分数（1-10）和评语
        """
        # 限制输入长度，避免上下文过长
        truncated = text[:4000] + ("..." if len(text) > 4000 else "")

        system_prompt = REPORT_EVALUATION_SYSTEM_PROMPT
        user_prompt = f"请评估以下研究报告：\n\n{truncated}"

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            content = response.content.strip()

            # 简单解析 JSON
            import json
            # 提取 JSON 部分
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 5))
                comment = data.get("comment", "")
            else:
                # 降级：提取数字
                numbers = re.findall(r"\b([1-9]|10)\b", content)
                score = float(numbers[0]) if numbers else 5.0
                comment = "无法解析 LLM 评语"
        except Exception as e:
            logger.error(f"LLM 评分失败：{e}")
            score = 5.0
            comment = "LLM 评分异常，使用默认分"

        # 确保分数在 1-10 之间
        score = max(1.0, min(10.0, score))
        return score, comment


if __name__ == "__main__":
    # 简单测试
    sample_report = """# 研究背景
本报告旨在评估最新的深度学习模型在自然语言处理任务中的表现。
## 相关文献
- 文献1
- 文献2
## 核心方法
我们采用了基于 Transformer 的架构，并引入了新的注意力机制。
```python
def example():
    print("Hello, World!")
```
## 优缺点
优点：性能提升，适用范围广。
缺点：计算资源需求较高。
## 参考文献
1. 文献1
2. 文献2
"""
    evaluator = ReportEvaluator()
    result = evaluator.evaluate(sample_report)
    print(result)
