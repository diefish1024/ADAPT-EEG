# src/utils/logger.py

import logging
from pathlib import Path
import datetime

from typing import Optional

def setup_logger(log_dir: str = 'results/logs', log_file_name: Optional[str] = None) -> logging.Logger:
    """
    设置项目专用的日志记录器。
    日志将同时输出到控制台和指定文件。

    Args:
        log_dir (str): 日志文件存储的目录。
        log_file_name (str, optional): 日志文件的名称。如果为 None ，则根据当前时间生成。

    Returns:
        logging.Logger: 配置好的日志记录器实例。
    """
    # 确保日志目录存在
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    if log_file_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"experiment_{timestamp}.log"
    
    log_file_path = log_path / log_file_name

    # 获取或创建 logger 实例
    # 如果已经存在同名 logger，则返回已有的，避免重复添加 handler
    logger = logging.getLogger('ADAPT_EEG_Logger')
    logger.setLevel(logging.INFO) # 设置最低日志级别

    if not logger.handlers:
        # 控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件 handler
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logger initialized. Logs will be saved to: {log_file_path}")

    return logger
