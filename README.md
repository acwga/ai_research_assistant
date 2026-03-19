# AI Research Assistant - 智能研究助手

## 📌 项目简介
一个基于 **ReAct Agent** 的研究助手，能够自动规划研究任务、搜索学术论文、分析文献、生成代码，并输出结构化研究报告。

## 🚀 核心技术
- **任务规划**：LLM将复杂问题分解为原子化子任务（3-8步）
- **ReAct Agent**：LangGraph实现的多轮工具调用框架
- **工具集成**：arXiv搜索、GitHub搜索、论文分析、代码生成
- **记忆机制**：滑动窗口+摘要的短期记忆，支持上下文感知
- **流式执行**：生成器模式实现执行过程实时反馈

## 🛠️ 技术栈
- **语言模型**：通义千问 qwen3-max / OpenAI
- **Agent框架**：LangGraph + ReAct模式
- **前端界面**：Streamlit（支持历史记录、流式展示）
- **工具封装**：arXiv API、PyGithub、自定义分析工具

## 📁 项目结构
```
ai-research-assistant/
├── src/
│ ├── app.py # Streamlit前端界面
│ ├── config.py # 配置文件
│ ├── prompts.py # 提示词模板
│ │
│ ├── agent/
│ │ ├── planner.py # 任务规划器
│ │ ├── react_agent.py # ReAct Agent实现
│ │ └── stream_executor.py # 流式执行器
│ │
│ └── tools/
│ ├── arxiv_search.py # arXiv论文搜索
│ ├── github_search.py # GitHub仓库搜索
│ ├── paper_analyzer.py # 论文分析
│ ├── code_generator.py # 代码生成
│ └── report_writer.py # 报告撰写
│
├── .env # 环境变量
└── requirements.txt # 项目依赖
```

## 💡 核心亮点
1. **自主规划**：Planner将模糊问题分解为可执行的原子任务
2. **短期记忆**：滑动窗口+摘要，让Agent记住研究进展
3. **流式输出**：执行过程实时展示，用户体验友好
4. **多工具协同**：支持论文搜索、代码生成、报告撰写全流程
5. **历史记录**：自动保存每次研究，支持回顾

## 🎯 应用场景
- 学术文献调研
- 技术方案预研
- 论文阅读辅助
- 代码示例生成