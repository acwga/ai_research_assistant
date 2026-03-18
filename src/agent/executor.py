"""
执行器 (Executor)
整合 Planner 和 ReAct Agent，执行完整的研究流程
"""
from typing import List, Dict, Any
from src.agent.planner import ResearchPlanner
from src.agent.react_agent import ReActAgent
from langchain_core.tools import BaseTool
from src.tools.report_writer import write_report
import time

class ResearchExecutor:
    """
    研究执行器
    """

    def __init__(self, tools: List[BaseTool], max_iterations: int = 15):
        self.planner = ResearchPlanner()
        self.tools = tools
        self.max_iterations = max_iterations
        self.agent = ReActAgent(tools=tools, max_iterations=max_iterations)
        self.results = {
            "papers": [],
            "github": [],
            "code_examples": [],
            "comparisons": []
        }
    
    def execute(self, user_query: str, verbose: bool = True) -> str:
        """
        执行完整研究流程
        
        Args:
            user_query: 用户研究问题
            verbose: 是否打印详细信息
            
        Returns:
            最终研究报告
        """
        if verbose:
            print("\n" + "="*60)
            print("🔬 AI研究助手开始工作")
            print("="*60)

            print("\n📋 步骤1：任务规划")
            print("-" * 30)
        
        # 清空Agent的历史
        self.agent.clear_history()
        
        steps = self.planner.plan(user_query)

        if verbose:
            for i, step in enumerate(steps, 1):
                print(f"   {i}. {step}")

            print("\n🚀 步骤2：执行任务")
            print("-" * 30)
        
        step_results = []
        for i, step in enumerate(steps, 1):
            if verbose:
                print(f"\n  执行步骤 {i}/{len(steps)}：{step}")

            result = self.agent.run(step)

            # 记录结果
            step_result = f"步骤{i}：{step}\n结果：{result[:200]}..."
            step_results.append(step_result)

            if verbose:
                print(f"  ✅ 步骤 {i} 完成")
                print(f"  结果摘要：{result[:200]}...")

            # 收集结果
            self._collect_results(result)

        if verbose:
            print("\n📝 步骤3：生成研究报告")
            print("-"*30)
        
        report = self._generate_report(user_query, step_results)

        if verbose:
            print("\n✅ 研究完成！")
            print("="*60)

        # 重置Agent状态
        self.agent = ReActAgent(tools=self.tools, max_iterations=self.max_iterations)
        self.clear_results()

        return report
    
    def _collect_results(self, result: str):
        """
        收集步骤结果，分类存储
        """
        if "论文" in result or "arxiv" in result or "paper" in result.lower():
            self.results["papers"].append(result)
        elif "github" in result or "仓库" in result or "代码" in result.lower():
            self.results["github"].append(result)
        elif "```" in result or "代码示例" in result:
            self.results["code_examples"].append(result)
        elif "对比" in result or "比较" in result:
            self.results["comparisons"].append(result)

    def _generate_report(self, user_query: str, step_results: List[str]) -> str:
        """
        生成最终研究报告
        """
        try:
            report = write_report.invoke({
                "topic": user_query,
                "research_data": self.results
            })

            return report
        
        except Exception as e:
            return f"""# {user_query} 研究报告

## 执行过程
{chr(10).join([f'- {r}' for r in step_results])}

## 收集资料统计
- 论文资料：{len(self.results['papers'])}篇
- GitHub项目：{len(self.results['github'])}个
- 代码示例：{len(self.results['code_examples'])}个
- 对比分析：{len(self.results['comparisons'])}项

## 说明
报告生成失败：{str(e)}
使用简单模板替代。
"""
    
    def clear_results(self):
        """
        清空结果
        """
        self.results = {
            "papers": [],
            "github": [],
            "code_examples": [],
            "comparisons": []
        }

def research(user_query: str, tools: List[BaseTool], verbose: bool = True) -> str:
    """
    执行研究流程的快捷函数
    """
    executor = ResearchExecutor(tools)
    return executor.execute(user_query, verbose)

if __name__ == "__main__":
    from src.tools.arxiv_search import search_arxiv
    from src.tools.github_search import search_github_repositories
    from src.tools.paper_analyzer import analyze_paper
    from src.tools.code_generator import generate_code

    tools = [search_arxiv, search_github_repositories, analyze_paper, generate_code, write_report]

    result = research("Transformer模型简介", tools)
    print("\n最终研究报告：")
    print(result)