import sys
import logging
import os

def setup_logging(log_file_path):
    try:
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Create file handler with error handling
        try:
            file_handler = logging.FileHandler(log_file_path)
        except PermissionError:
            print(f"Warning: Cannot write to {log_file_path}. Logging to console only.")
            return logging.getLogger()

        file_handler.setLevel(logging.INFO)

        # ...existing code...

        return logger
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        return logging.getLogger()
