"""
GitHub 仓库搜索工具
使用 PyGithub 库实现简单的仓库搜索功能
"""
import os
from typing import Optional
from github import Github, GithubException
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def search_github_repositories(
    query: str,
    max_results: int = 5,
    sort: str = "stars",
    order: str = "desc",
    language: Optional[str] = None
) -> str:
    """
    在 GitHub 上搜索代码仓库
    Args:
        query (str): 搜索关键词
        max_results (int): 返回的最大仓库数量，默认为 5
        sort (str): 排序方式，默认为 "stars"
        order (str): 排序顺序，默认为 "desc"
        language (Optional[str]): 可选的编程语言过滤
    Returns:
        str: 仓库名称、star数、描述、链接等信息的文本汇总
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return "GitHub 访问令牌未设置，请在环境变量中配置 GITHUB_TOKEN。"
    
    try:
        g = Github(token)
        search_query = query
        if language:
            search_query += f" language:{language}"
        
        repos = g.search_repositories(
            query=search_query,
            sort=sort,
            order=order
        )

        results = []
        count = 0
        for repo in repos:
            if count >= max_results:
                break

            description = (repo.description or "无描述")[:200]
            if len(description) == 200:
                description += "..."

            item = (
                f"[{repo.full_name}]"
                f"\n⭐ {repo.stargazers_count:,}  |  🍴 {repo.forks_count:,}"
                f"\n{description}"
                f"\n{repo.html_url}"
            )
            results.append(item)
            count += 1
        
        if not results:
            return "未找到匹配的仓库。"
        
        header = f"找到 {repos.totalCount} 个匹配的仓库，以下是前 {count} 个（按 {sort} {order} 排序）：\n\n"
        return header + "\n\n".join(results)
    
    except GithubException as e:
        if e.status == 403:
            return "GitHub API 限流（403）。请稍后再试，或检查 GITHUB_TOKEN 是否有效。"
        return f"GitHub API 错误：{str(e)}"
    except Exception as e:
        return f"执行出错：{str(e)}"
    
if __name__ == "__main__":
    result = search_github_repositories.invoke({
        "query": "LangGraph ReAct agent",
        "max_results": 3,
        "language": "python"
    })
    print(result)