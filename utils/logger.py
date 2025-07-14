from loguru import logger
import sys

# Console log
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")
# File log daily rotation
logger.add("logs/{time:YYYY-MM-DD}.log", rotation="00:00", retention="7 days", level="DEBUG")
