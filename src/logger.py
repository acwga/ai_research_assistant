import logging
import os
from datetime import datetime

# 创建 logs 目录
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 生成日志文件名（按天分割）
log_filename = os.path.join(LOG_DIR, f"research_assistant_{datetime.now().strftime('%Y%m%d')}.log")

# 配置根日志器
logger = logging.getLogger("research_assistant")
logger.setLevel(logging.DEBUG)

# 文件处理器：记录所有级别
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

# 控制台处理器：只记录 INFO 及以上（避免刷屏）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 设置格式
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 提供一个快捷函数
def get_logger(name=None):
    if name:
        return logging.getLogger(f"research_assistant.{name}")
    return logger