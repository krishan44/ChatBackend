#!/usr/bin/env python3
"""
Setup script for the Education Chatbot.
This script helps install necessary dependencies and sets up the environment.
"""
import subprocess
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is 3.7 or higher"""
    if sys.version_info < (3, 7):
        logger.error("Python version must be 3.7 or higher")
        return False
    return True

def install_dependencies():
    """Install required Python packages"""
    packages = [
        "flask",
        "flask-cors",
        "pandas",
        "openpyxl",  # For Excel file support
        "scikit-learn",
        "numpy",
        "spacy"
    ]

    logger.info("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        for package in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info("Python dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {str(e)}")
        return False

def install_spacy_model():
    """Install spaCy language model"""
    logger.info("Installing spaCy language model (en_core_web_sm)...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        logger.info("spaCy language model installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install spaCy model: {str(e)}")
        logger.info("You can manually install it by running: python -m spacy download en_core_web_sm")
        return False

def setup_dataset_folder():
    """Create Dataset folder if it doesn't exist"""
    dataset_path = os.path.join(os.getcwd(), 'Dataset')
    if not os.path.exists(dataset_path):
        logger.info("Creating Dataset directory...")
        os.makedirs(dataset_path)
        logger.info(f"Dataset directory created at {dataset_path}")
    else:
        logger.info(f"Dataset directory already exists at {dataset_path}")
    return True

def main():
    """Main setup function"""
    logger.info("Starting Education Chatbot setup...")
    
    if not check_python_version():
        return 1
    
    if not install_dependencies():
        logger.warning("Some dependencies could not be installed. You may need to install them manually.")
    
    if not install_spacy_model():
        logger.warning("The spaCy model could not be installed automatically.")
    
    setup_dataset_folder()
    
    logger.info("""
Setup completed! You can now run the chatbot using:
    python app.py

If you encountered any issues with the spaCy model, try:
    python -m spacy download en_core_web_sm
    """)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
