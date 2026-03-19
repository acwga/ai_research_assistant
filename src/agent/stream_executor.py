"""
执行器：整合 Planner 和 ReAct Agent，协调整个研究流程
支持逐步执行，用于流式输出。
"""
from src.agent.planner import ResearchPlanner
from src.agent.react_agent import ReActAgent, create_agent
from src.tools.arxiv_search import search_arxiv
from src.tools.github_search import search_github_repositories
from src.tools.paper_analyzer import analyze_paper, compare_papers
from src.tools.code_generator import generate_code, explain_code
from src.tools.report_writer import write_report, summarize_findings
from typing import Generator, Dict, Any

class StreamingResearchExecutor:
    """
    支持流式输出的研究任务执行器
    每次调用 execute 返回一个生成器，逐步产出：
      - 步骤列表
      - 每个步骤的执行结果
      - 最终报告
    """

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

    def execute(self, user_query: str) -> Generator[Dict[str, Any], None, None]:
        """
        执行完整研究流程，逐步产出：
        - 步骤列表: {"type": "steps", "steps": list}
        - 每个步骤的结果: {"type": "step_result", "index": int, "step": str, "result": str}
        - 最终报告: {"type": "report", "report": str}
        """
        # Step 1: 任务分解
        steps = self.planner.plan(user_query)
        yield {"type": "steps", "steps": steps}

        # Step 2: 顺序执行子任务，收集结果
        research_data = {
            "papers": [],
            "github": [],
            "code_examples": [],
            "comparisons": []
        }

        for idx, step in enumerate(steps, 1):
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

            yield {
                "type": "step_result",
                "index": idx,
                "step": step,
                "result": result
            }

        # Step 3: 生成最终报告
        final_report = write_report.invoke({
            "topic": user_query,
            "research_data": research_data
        })

        yield {"type": "report", "report": final_report}

    def clear_history(self):
        """清空 Agent 历史"""
        self.agent.clear_history()


if __name__ == "__main__":
    # 测试流式执行
    executor = StreamingResearchExecutor()
    query = "Transformer 和 BERT 在文本分类任务中的对比"
    for output in executor.execute(query):
        if output["type"] == "steps":
            print("\n📋 规划步骤:")
            for i, step in enumerate(output["steps"], 1):
                print(f"   {i}. {step}")
        elif output["type"] == "step_result":
            print(f"\n⚡ 步骤 {output['index']} 完成:")
            print(f"   结果摘要: {output['result'][:200]}...")
        elif output["type"] == "report":
            print("\n📝 最终报告:")
            print(output["report"])