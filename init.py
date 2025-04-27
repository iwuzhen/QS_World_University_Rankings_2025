from dotenv import load_dotenv

load_dotenv()

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(".")))

import logging

# 配置日志记录器
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
