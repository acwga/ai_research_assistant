"""
执行器：整合 Planner 和 ReAct Agent，协调整个研究流程
"""
from src.agent.planner import ResearchPlanner
from src.agent.react_agent import ReActAgent, create_agent
from src.tools.arxiv_search import search_arxiv
from src.tools.github_search import search_github_repositories
from src.tools.paper_analyzer import analyze_paper, compare_papers
from src.tools.code_generator import generate_code, explain_code
from src.tools.report_writer import write_report, summarize_findings

class ResearchExecutor:
    """研究任务执行器"""

    def __init__(self):
        # 初始化所有工具
        self.tools = [
            search_arxiv,
            search_github_repositories,
            analyze_paper,
            compare_papers,
            generate_code,
            explain_code,
            # write_report,
            summarize_findings,
        ]
        # 创建 ReAct Agent（不限制迭代次数，由 Planner 分解步骤控制）
        self.agent = create_agent(self.tools)
        self.planner = ResearchPlanner()

    def execute(self, user_query: str) -> str:
        """
        执行完整研究流程：
        1. Planner 分解任务
        2. Agent 顺序执行每个子任务
        3. 收集结果并生成最终报告
        """
        print(f"\n📋 开始研究: {user_query}")
        print("=" * 60)

        # Step 1: 任务分解
        print("🔍 步骤1: 任务规划中...")
        steps = self.planner.plan(user_query)
        print(f"规划完成，共 {len(steps)} 个子任务:")
        for i, step in enumerate(steps, 1):
            print(f"   {i}. {step}")

        # Step 2: 顺序执行子任务，收集结果
        research_data = {
            "papers": [],
            "github": [],
            "code_examples": [],
            "comparisons": []
        }

        for idx, step in enumerate(steps, 1):
            print(f"\n⚡ 执行子任务 {idx}/{len(steps)}: {step}")
            print("-" * 40)

            # 每个子任务使用独立的 Agent 实例（或清空历史）避免上下文干扰
            # 这里我们使用同一个 agent 但每次调用 run_step 会新建临时历史
            result = self.agent.run_step(step)

            # 根据步骤描述分类存储结果（可根据实际需要优化分类逻辑）
            step_lower = step.lower()
            if "论文" in step_lower or "arxiv" in step_lower or "文献" in step_lower:
                research_data["papers"].append(result)
            elif "github" in step_lower or "仓库" in step_lower or "代码库" in step_lower:
                research_data["github"].append(result)
            elif "代码" in step_lower or "生成" in step_lower and "代码" in step_lower:
                research_data["code_examples"].append(result)
            elif "对比" in step_lower or "比较" in step_lower:
                research_data["comparisons"].append(result)
            else:
                # 默认归入杂项，报告生成时会用到所有结果
                # 这里简单处理，将无法分类的结果附加到某个部分（例如 papers）
                research_data["papers"].append(result)

            print(f"✅ 子任务完成")

        # Step 3: 生成最终报告
        print("\n📝 步骤3: 生成研究报告...")
        final_report = write_report.invoke({
            "topic": user_query,
            "research_data": research_data
        })

        print("\n" + "=" * 60)
        print("🎉 研究完成！")
        return final_report

    def clear_history(self):
        """清空 Agent 历史"""
        self.agent.clear_history()


if __name__ == "__main__":
    # 测试执行器
    executor = ResearchExecutor()
    query = "Transformer 和 BERT 在文本分类任务中的对比"
    report = executor.execute(query)
    print("\n" + report)