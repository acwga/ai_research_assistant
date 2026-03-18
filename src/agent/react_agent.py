"""
ReAct Agent - 使用 LangGraph 框架实现
"""
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from src.config import OLLAMA_BASE_URL, MODEL_NAME
from src.prompts import REACT_SYSTEM_PROMPT, get_tools_description
from openai import OpenAI
import json

client = OpenAI(
    api_key="ollama",
    base_url=OLLAMA_BASE_URL
)

# 定义状态
class AgentState(TypedDict):
    messages: List[BaseMessage]     # 对话历史
    user_query: str     # 当前用户查询
    iteration: int      # 当前迭代次数

class ReActAgent:
    """
    ReAct Agent类
    """

    def __init__(self, tools: List[BaseTool], max_iterations: int = 10):
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.llm = client
        self.tools_description = get_tools_description(tools)
        self.system_prompt = REACT_SYSTEM_PROMPT.format(
            tools_description=self.tools_description
        )
        self.graph = self._build_graph()
        self.conversation_history = []

    def _call_llm(self, state: AgentState):
        """
        调用LLM获取下一步动作
        """
        messages = state["messages"]

        # 检查迭代次数
        if state["iteration"] >= self.max_iterations:
            return {
                "messages": messages + [AIMessage(content="已达到最大迭代次数，结束任务。")],
                "iteration": state["iteration"] + 1
            }
        
        # 调用LLM
        response = self.llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": self.system_prompt},
                *[{"role": "user" if isinstance(m, HumanMessage) else "assistant",
                 "content": m.content}
                 for m in messages]
            ],
            temperature=0.3,
            max_tokens=500
        )

        ai_message = response.choices[0].message.content
        
        return {
            "messages": messages + [AIMessage(content=ai_message)],
            "iteration": state["iteration"] + 1
        }
    
    def _execute_tool(self, state: AgentState):
        """
        执行工具调用
        """
        messages = state["messages"]
        last_message = messages[-1].content

        # 解析Action
        action = self._parse_action(last_message)
        if not action:
            return {
                "messages": messages + [HumanMessage(content="格式错误，请按格式输出")],
                "iteration": state["iteration"]
            }
        
        tool_name = action["name"]
        tool_input = action["input"]

        # 执行工具
        if tool_name not in self.tools:
            result = f"错误：工具 '{tool_name}' 不存在"
        else:
            try:
                tool = self.tools[tool_name]
                if isinstance(tool_input, dict):
                    result = tool.invoke(tool_input)
                else:
                    try:
                        result = tool.invoke(tool_input)
                    except:
                        result = tool.invoke({"input": tool_input})
            except Exception as e:
                result = f"工具执行出错：{str(e)}"
        
        return {
            "messages": messages + [HumanMessage(content=f"Observation: {result}")],
            "iteration": state["iteration"]
        }

    def _parse_action(self, text: str) -> dict:
        """
        解析Action和Action Input
        """
        try:
            lines = text.strip().split("\n")
            action = None
            action_input = None

            for line in lines:
                if line.startswith("Action:"):
                    action = line.replace("Action:", "").strip()
                elif line.startswith("Action Input:"):
                    input_str = line.replace("Action Input:", "").strip()
                    try:
                        action_input = json.loads(input_str)
                    except:
                        action_input = input_str
            
            if action and action_input is not None:
                return {"name": action, "input": action_input}
            return None
        
        except:
            return None
        
    def _should_continue(self, state: AgentState):
        """
        判断是否继续迭代
        """
        last_message = state["messages"][-1].content
        
        if "Final Answer:" in last_message:
            return "end"
        
        if state["iteration"] >= self.max_iterations:
            return "end"
        
        return "continue"

    def _build_graph(self):
        """
        构建状态图
        """
        # 创建图
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("llm", self._call_llm)
        workflow.add_node("tool", self._execute_tool)

        # 添加边
        workflow.add_edge(START, "llm")
        workflow.add_conditional_edges(
            "llm",
            self._should_continue,
            {
                "continue": "tool",
                "end": END
            }
        )
        workflow.add_edge("tool", "llm")

        return workflow.compile()
    
    def run(self, user_query: str, context: str = "") -> str:
        """
        运行ReAct Agent
        """
        print(f"\n开始研究：{user_query}")

        # 将新查询加入历史
        self.conversation_history.append(HumanMessage(content=user_query))

        # 初始化状态
        initial_state = {
            "messages": self.conversation_history.copy(),
            "user_query": user_query,
            "iteration": 0
        }

        # 执行状态图
        final_state = None
        for state in self.graph.stream(initial_state):
            final_state = state

        # 提取最终答案并保存到历史
        if final_state and "llm" in final_state:
            for msg in final_state["llm"]["messages"]:
                if msg not in self.conversation_history:
                    self.conversation_history.append(msg)
            
            for msg in reversed(final_state["llm"]["messages"]):
                if "Final Answer:" in msg.content:
                    return msg.content.split("Final Answer:")[-1].strip()
            
            last_msg = final_state["llm"]["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                return last_msg.content
        
        return "未能生成最终答案"
    
    def clear_history(self):
        """
        清空对话历史
        """
        self.conversation_history = []
    
def create_agent(tools: List[BaseTool]) -> ReActAgent:
    """
    创建Agent
    """
    return ReActAgent(tools)
    
if __name__ == "__main__":
    from src.tools.arxiv_search import search_arxiv
    from src.tools.paper_analyzer import analyze_paper

    tools = [search_arxiv, analyze_paper]
    agent = create_agent(tools)

    result = agent.run("什么是Transformer？")
    print(f"\n最终结果：{result}")