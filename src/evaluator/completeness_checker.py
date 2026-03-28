"""
完整性检查器
评估研究数据中各部分是否充足（论文、GitHub、代码、对比分析）
"""
from typing import Dict, List, Any
from src.logger import get_logger

logger = get_logger("completeness_checker")


class CompletenessChecker:
    """研究数据完整性检查器"""

    def __init__(self):
        # 各部分满分和评分规则
        self.max_score = 10.0
        self.section_weights = {
            "papers": 2.5,
            "github": 2.5,
            "code_examples": 2.5,
            "comparisons": 2.5
        }

    def evaluate(self, research_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        评估研究数据的完整性

        Args:
            research_data: 包含论文、GitHub、代码示例、对比分析的字典
                          格式：{"papers": [...], "github": [...], "code_examples": [...], "comparisons": [...]}

        Returns:
            包含总分、各子项得分、详细统计和建议的字典
        """
        logger.info("开始完整性检查")

        # 提取各部分列表
        papers = research_data.get("papers", [])
        github = research_data.get("github", [])
        code_examples = research_data.get("code_examples", [])
        comparisons = research_data.get("comparisons", [])

        # 统计数量
        stats = {
            "papers": len(papers),
            "github": len(github),
            "code_examples": len(code_examples),
            "comparisons": len(comparisons)
        }
        logger.debug(f"统计数据: {stats}")

        # 计算各单项得分（0~权重值）
        paper_score = self._score_by_count(stats["papers"], self.section_weights["papers"])
        github_score = self._score_by_count(stats["github"], self.section_weights["github"])
        code_score = self._score_by_count(stats["code_examples"], self.section_weights["code_examples"])
        comp_score = self._score_by_count(stats["comparisons"], self.section_weights["comparisons"])

        total_score = paper_score + github_score + code_score + comp_score

        # 生成改进建议
        suggestions = []
        if stats["papers"] == 0:
            suggestions.append("未找到任何相关论文，请增加论文搜索或调整搜索关键词。")
        elif stats["papers"] == 1:
            suggestions.append("仅找到1篇论文，建议搜索更多相关文献以获得更全面的综述。")

        if stats["github"] == 0:
            suggestions.append("未找到任何GitHub仓库，建议增加代码库搜索，或检查关键词是否准确。")
        elif stats["github"] == 1:
            suggestions.append("仅找到1个GitHub仓库，可尝试搜索更多实现，以获得多样化示例。")

        if stats["code_examples"] == 0:
            suggestions.append("未生成任何代码示例，建议在步骤中增加代码生成任务，或检查代码生成工具是否可用。")
        elif stats["code_examples"] == 1:
            suggestions.append("代码示例较少，可增加更多技术或场景的代码生成。")

        if stats["comparisons"] == 0:
            suggestions.append("未进行任何对比分析，建议增加对比步骤，突出不同方法的优劣。")
        elif stats["comparisons"] == 1:
            suggestions.append("仅进行了一组对比，可增加更多维度的对比分析（如性能、资源消耗等）。")

        # 如果没有缺失，给出表扬
        if not suggestions:
            suggestions.append("研究数据非常完整，覆盖了论文、GitHub、代码示例和对比分析，质量良好。")

        result = {
            "total_score": round(total_score, 2),
            "max_score": self.max_score,
            "scores": {
                "papers": round(paper_score, 2),
                "github": round(github_score, 2),
                "code_examples": round(code_score, 2),
                "comparisons": round(comp_score, 2)
            },
            "stats": stats,
            "suggestions": suggestions
        }

        logger.info(f"完整性检查完成，总分: {result['total_score']}/{self.max_score}")
        return result

    def _score_by_count(self, count: int, max_score: float) -> float:
        """
        根据数量计算得分
        规则：
            count=0 -> 0
            count=1 -> max_score * 0.6 (1.5分)
            count=2 -> max_score * 0.8 (2分)
            count>=3 -> max_score (2.5分)
        """
        if count == 0:
            return 0.0
        elif count == 1:
            return max_score * 0.6
        elif count == 2:
            return max_score * 0.8
        else:
            return max_score


if __name__ == "__main__":
    # 简单测试
    test_data = {
        "papers": ["paper1", "paper2"],
        "github": ["repo1"],
        "code_examples": ["code1"],
        "comparisons": []
    }
    checker = CompletenessChecker()
    result = checker.evaluate(test_data)
    print(result)