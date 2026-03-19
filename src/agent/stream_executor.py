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
from openai import OpenAI
from src.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME
from src.prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT_TEMPLATE
from collections import deque

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

        self.llm = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL
        )
        self.memory = deque(maxlen=5)

    def _summarize_memory(self, step: str, result: str) -> str:
        """
        使用 LLM 将步骤结果压缩为一句摘要
        """
        prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
            step=step,
            result=result
        )
        try：
            response = self.llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            # 如果摘要失败，返回一个简单的截断
            return result[:100] + "..."


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

        # 清空记忆（每次新研究重置）
        self.memory.clear()

        # Step 2: 顺序执行子任务
        for idx, step in enumerate(steps, 1):
            # 构建增强提示词：注入研究主题和已有记忆
            enhanced_step = f"研究主题：{user_query}\n"
            if self.memory:
                memory_text = "\n".join([f"- 步骤{i+1}：{mem}" for i, mem in enumerate(self.memory)])
                enhanced_step += f"\n已有研究进展：\n{memory_text}\n"
            enhanced_step += f"\n请完成以下任务：{step}"

            result = self.agent.run_step(enhanced_step)

            # 生成摘要并存入记忆
            summary = self._summarize_memory(step, result)
            self.memory.append(summary)

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