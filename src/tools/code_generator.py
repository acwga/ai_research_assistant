"""
代码生成工具
根据论文描述或技术需求生成示例代码
"""
from langchain_core.tools import tool
from src.config import OLLAMA_BASE_URL, MODEL_NAME
from openai import OpenAI
from src.prompts import (
    CODE_GENERATION_SYSTEM_PROMPT_TEMPLATE,
    CODE_GENERATION_USER_PROMPT_TEMPLATE,
    CODE_EXPLAIN_SYSTEM_PROMPT,
    CODE_EXPLAIN_USER_PROMPT_TEMPLATE
)

client = OpenAI(
    api_key="ollama",
    base_url=OLLAMA_BASE_URL
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
    prompt = CODE_GENERATION_USER_PROMPT_TEMPLATE.format(
        technique=technique,
        language=language,
        context=context,
        context_text=f'额外要求：{context}' if context else ''
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": CODE_GENERATION_SYSTEM_PROMPT_TEMPLATE.format(language=language)},
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
    prompt = CODE_EXPLAIN_USER_PROMPT_TEMPLATE.format(
        language=language,
        code=code
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": CODE_EXPLAIN_SYSTEM_PROMPT},
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