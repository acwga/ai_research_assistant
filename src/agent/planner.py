"""
任务规划器 (Planner)
"""
from langchain_core.tools import tool
from src.config import OLLAMA_BASE_URL, MODEL_NAME
from src.prompts import PLANNER_PROMPT
from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url=OLLAMA_BASE_URL
)

class ResearchPlanner:
    """
    研究任务规划器
    """
    def __init__(self):

        self.llm = client

    def plan(self, user_query: str) -> list:
        """
    将用户查询分解为子任务列表
    
    Args:
        user_query: 用户的研究问题
        
    Returns:
        子任务列表，格式：["步骤1", "步骤2", ...]
    """
        # 构建提示词
        prompt = PLANNER_PROMPT.format(user_query=user_query)

        # 生成计划
        try:
            response = self.llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个专业的任务规划专家，只输出编号列表。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            plan_text = response.choices[0].message.content.strip()
            # 解析计划文本
            steps = self._parse_plan(plan_text, user_query)
            if not self.validate_plan(steps):
                print("生成的计划不合理，使用默认计划")
                steps = self._get_default_plan(user_query)

            return steps
        
        except Exception as e:
            print(f"规划失败：{str(e)}")
            # 返回默认计划
            return self._get_default_plan(user_query)
        
    def _parse_plan(self, plan_text: str, user_query: str) -> list:
        """
        解析LLM返回的计划文本，提取步骤列表
        """
        steps = []
        for line in plan_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # 移除编号和开头符号
                step = line.split('.', 1)[-1] if "." in line else line
                step = step.strip('- •').strip()
                if step:
                    steps.append(step)
        
        # 如果解析失败，返回默认计划
        if not steps:
            return self._get_default_plan(user_query)
        
        return steps
    
    def _get_default_plan(self, user_query: str) -> list:
        """
        获取默认计划
        """
        return [
            f"搜索与 '{user_query}' 相关的学术论文",
            f"搜索与 '{user_query}' 相关的GitHub仓库",
            "分析找到的论文和代码",
            "对比不同方法的优缺点",
            "生成代码示例",
            "撰写最终研究报告"
        ]
    
    def validate_plan(self, steps: list) -> bool:
        """
        验证生成的计划是否合理
        """
        if not steps:
            return False
        if len(steps) < 2:
            return False
        if len(steps) > 10:
            return False
        return True
    
@tool
def create_research_plan(user_query: str) -> str:
    """
    创建研究计划工具

    Args:
        user_query: 用户的研究问题
    
    Returns:
        格式化的计划文本
    """
    planner = ResearchPlanner()
    steps = planner.plan(user_query)

    # 格式化输出
    result = f"## 📋 研究计划：{user_query}\n\n"
    for i, step in enumerate(steps, 1):
        result += f"{i}. {step}\n"

    return result

if __name__ == "__main__":
    print("="*50)
    print("测试：任务规划器")
    print("="*50)
    
    test_queries = [
        "Transformer和BERT在NLP任务中的性能对比",
        "LoRA微调技术的最新进展",
        "如何用LangGraph构建ReAct Agent"
    ]
    
    planner = ResearchPlanner()
    
    for query in test_queries:
        print(f"\n📝 用户问题：{query}")
        print("-"*30)
        steps = planner.plan(query)
        for i, step in enumerate(steps, 1):
            print(f"{i}. {step}")