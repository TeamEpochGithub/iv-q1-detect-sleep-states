import os
import logging

# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level (INFO, WARNING, ERROR, etc.)
logger.setLevel(logging.DEBUG)

# Create the "logs" folder if it doesn't exist
logs_folder = "logs"
os.makedirs(logs_folder, exist_ok=True)

# Define the log file path
log_file_path = os.path.join(logs_folder, "app.log")

# Create a file handler
file_handler = logging.FileHandler(log_file_path)

# Set the log message format
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)
