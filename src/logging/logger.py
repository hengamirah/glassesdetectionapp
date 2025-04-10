import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(config):
    """Set up logging based on configuration"""
    log_level = getattr(logging, config.get('level', 'INFO').upper())
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create logger
    logger = logging.getLogger('glasses_detection')
    logger.setLevel(log_level)
    logger.handlers = []  # Clear any existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if 'file' in config:
        # Ensure log directory exists
        log_dir = os.path.dirname(config['file'])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = RotatingFileHandler(
            config['file'],
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info("Logging setup complete")
    
    return logger