"""
综合评估器
整合报告质量评估和完整性检查，生成综合评分和改进建议
"""
from typing import Dict, Any
from src.evaluator.report_evaluator import ReportEvaluator
from src.evaluator.completeness_checker import CompletenessChecker
from src.logger import get_logger

logger = get_logger("evaluator")


class ResearchEvaluator:
    """综合评估器"""

    def __init__(self, report_model: str = "qwen2.5:7b"):
        self.report_evaluator = ReportEvaluator(model=report_model)
        self.completeness_checker = CompletenessChecker()

    def evaluate(self, report_text: str, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合评估研究报告和研究过程

        Args:
            report_text: 最终报告文本
            research_data: 研究数据字典，格式如 {"papers": [], "github": [], "code_examples": [], "comparisons": []}

        Returns:
            综合评估结果，包含各子模块得分、总得分和建议
        """
        logger.info("开始综合评估")

        # 1. 报告质量评估
        logger.info("进行报告质量评估...")
        report_result = self.report_evaluator.evaluate(report_text)

        # 2. 数据完整性评估
        logger.info("进行数据完整性评估...")
        completeness_result = self.completeness_checker.evaluate(research_data)

        # 3. 综合得分（简单平均，两者满分都是10）
        report_score = report_result.get("total_score", 0)
        completeness_score = completeness_result.get("total_score", 0)
        overall_score = (report_score + completeness_score) / 2

        # 4. 合并建议
        suggestions = []
        # 从报告评估中提取 LLM 评语（如果有）
        llm_comment = report_result.get("llm_based", {}).get("comment", "")
        if llm_comment:
            suggestions.append(llm_comment)
        # 加入完整性检查的建议
        suggestions.extend(completeness_result.get("suggestions", []))
        # 去重并保留非空
        unique_suggestions = list(dict.fromkeys([s for s in suggestions if s.strip()]))

        result = {
            "overall_score": round(overall_score, 2),
            "report_quality": report_result,
            "completeness": completeness_result,
            "suggestions": unique_suggestions[:5]  # 最多5条建议
        }

        logger.info(f"综合评估完成，总分: {result['overall_score']}")
        return result


def evaluate_research(report_text: str, research_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    便捷函数：一站式评估研究报告和研究数据

    Args:
        report_text: 最终报告文本
        research_data: 研究数据字典

    Returns:
        综合评估结果
    """
    evaluator = ResearchEvaluator()
    return evaluator.evaluate(report_text, research_data)


if __name__ == "__main__":
    # 简单测试
    sample_report = "这是一个关于人工智能研究的报告，内容详实，结构清晰，但缺乏创新点。"
    sample_data = {
        "papers": ["Paper 1", "Paper 2"],
        "github": ["Repo 1"],
        "code_examples": ["Example 1"],
        "comparisons": ["Comparison 1"]
    }
    result = evaluate_research(sample_report, sample_data)
    print(result)