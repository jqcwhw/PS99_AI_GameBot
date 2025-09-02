"""
Logging Configuration for AI Game Bot
Provides centralized logging setup and configuration
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
import time

def setup_logger(name: str = 'GameBot', level: int = logging.INFO) -> logging.Logger:
    """
    Setup centralized logger for the application
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler for detailed logs
    log_file = logs_dir / "gamebot.log"
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_log_file = logs_dir / "errors.log"
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    logger.info(f"Logger '{name}' initialized")
    return logger

class LogBuffer:
    """Buffer for storing recent log entries for web interface"""
    
    def __init__(self, max_size: int = 100):
        self.buffer = []
        self.max_size = max_size
    
    def add_entry(self, level: str, message: str, timestamp: Optional[float] = None):
        """Add a log entry to the buffer"""
        if timestamp is None:
            timestamp = time.time()
        
        entry = {
            'level': level,
            'message': message,
            'timestamp': timestamp,
            'formatted_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        }
        
        self.buffer.append(entry)
        
        # Keep buffer size under limit
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
    
    def get_entries(self, level_filter: Optional[str] = None, limit: Optional[int] = None) -> list:
        """Get log entries, optionally filtered by level"""
        entries = self.buffer
        
        if level_filter:
            entries = [entry for entry in entries if entry['level'] == level_filter]
        
        if limit:
            entries = entries[-limit:]
        
        return entries
    
    def clear(self):
        """Clear all buffered entries"""
        self.buffer.clear()

class BufferedHandler(logging.Handler):
    """Custom handler that stores logs in a buffer for web interface"""
    
    def __init__(self, buffer: LogBuffer):
        super().__init__()
        self.buffer = buffer
    
    def emit(self, record):
        """Emit a log record to the buffer"""
        try:
            message = self.format(record)
            self.buffer.add_entry(record.levelname, message, record.created)
        except Exception:
            self.handleError(record)

def setup_web_logger(buffer_size: int = 100) -> tuple[logging.Logger, LogBuffer]:
    """
    Setup logger with web interface buffer
    
    Args:
        buffer_size: Maximum number of log entries to buffer
        
    Returns:
        Tuple of (logger, log_buffer)
    """
    logger = setup_logger()
    
    # Create buffer and handler
    log_buffer = LogBuffer(buffer_size)
    buffer_handler = BufferedHandler(log_buffer)
    buffer_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(buffer_handler)
    
    return logger, log_buffer

def configure_third_party_loggers():
    """Configure third-party library loggers to reduce noise"""
    # Reduce verbosity of common third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Configure third-party loggers when module is imported
configure_third_party_loggers()
