"""
代码生成工具
根据论文描述或技术需求生成示例代码
"""
from langchain_core.tools import tool
from src.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, MODEL_NAME
from openai import OpenAI

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL
)

@tool
def generate_code(technique: str, language: str = "python", context: str = "") -> str:
    """
    根据技术描述生成示例代码
    
    Args:
        technique: 要生成代码的技术名称或描述（如："Transformer注意力机制", "LoRA微调"）
        language: 编程语言，默认python
        context: 额外的上下文信息，如具体需求或约束
    
    Returns:
        生成的代码示例和说明
    """
    prompt = f"""
    请为以下技术生成简洁的示例代码：
    
    技术：{technique}
    编程语言：{language}
    {f'额外要求：{context}' if context else ''}
    
    要求：
    1. 代码要简洁明了，突出核心概念
    2. 添加必要的注释解释关键步骤
    3. 如果代码需要依赖，说明需要安装的包
    4. 提供一个简单的使用示例
    
    请按以下格式输出：
    
    ## {technique} 代码示例
    
    ### 依赖安装
    ```bash
    [需要的安装命令]
    ```
    
    ### 核心代码
    ```{language}
    [代码实现]
    ```
    
    ### 使用示例
    ```{language}
    [如何使用这段代码的示例]
    ```
    
    ### 说明
    [代码的关键点解释和使用注意事项]
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"你是一个专业的{language}工程师，擅长生成清晰、可运行的示例代码。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1200
        )

        return response.choices[0].message.content
    
    except Exception as e:
        return f"代码生成失败：{str(e)}"
    
@tool
def explain_code(code: str, language: str = "python") -> str:
    """
    解释代码的功能和原理
    
    Args:
        code: 要解释的代码
        language: 编程语言
    
    Returns:
        代码解释说明
    """
    prompt = f"""
    请解释以下{language}代码的功能和原理：
    
    ```{language}
    {code}
    ```
    
    请按以下格式输出：
    
    ## 代码功能
    [这段代码的主要功能]
    
    ## 逐行解释
    [关键行的解释]
    
    ## 核心原理
    [使用的技术原理]
    
    ## 可能的改进
    [如果有，可以如何优化]
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个耐心的编程导师，擅长解释代码。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )

        return response.choices[0].message.content
    
    except Exception as e:
        return f"代码解释失败：{str(e)}"
    
if __name__ == "__main__":
    print("="*50)
    print("测试1：生成Transformer注意力代码")
    print("="*50)
    result1 = generate_code.invoke({
        "technique": "Transformer的多头注意力机制",
        "language": "python",
        "context": "使用PyTorch实现，要包含缩放点积注意力"
    })
    print(result1)
    
    print("\n" + "="*50)
    print("测试2：解释一段代码")
    print("="*50)
    test_code = """
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        
        return torch.matmul(attention, V)
"""
    result2 = explain_code.invoke({
        "code": test_code,
        "language": "python"
    })
    print(result2)