import logging
import os
from logging import handlers


def get_logger(LOG_ROOT, level=logging.DEBUG, back_count=0,cmd_stream=False):
    """
    :brief  日志记录
    :param log_filename:
    :param level:
    :param back_count:
    :return: logger
    """
    logger = logging.getLogger("logger.log")
    logger.setLevel(level)
    log_path = os.path.join(LOG_ROOT, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file_path = os.path.join(log_path, "logger.log")
    # log输出格式
    formatter = logging.Formatter('%(asctime)s:\n %(message)s')
    # 输出到文件
    fh = logging.handlers.TimedRotatingFileHandler(
        filename=log_file_path,
        backupCount=back_count,
        encoding='utf-8')
    fh.setLevel(level)
    # 添加到logger对象里
    logger.addHandler(fh)
    # 输出到控制台
    if cmd_stream:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    logger = get_logger("./",cmd_stream=True)
    logger.debug("debug test")
    logger.info("info test")
    logger.error("error test")