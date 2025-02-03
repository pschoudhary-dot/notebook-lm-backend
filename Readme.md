# Project Overview

This project is a clone of Google's Notebook LM, offering similar features for document processing and chat functionality. It allows users to upload various document types, including PDF, CSV, images, video, audio, URLs, Docs, and Excel files. The uploaded data can be used to chat with the system, similar to the RAAF app. Additionally, users have the option to create transcripts for podcasts.

# Folder Structure

The project's folder structure is as follows:

* `_pycache_`: Python cache files
* `audio`: Audio files
* `docs`: Document files
* `transcripts`: Transcripts of podcasts
* `venv`: Virtual environment for Python
* `O env`: Environment configuration files
* `.gitignore`: Git ignore file
* `document_processor.log`: Log file for document processing
* `main.py`: Main application file
* `models.json`: JSON file containing model data
* `models.py`: Python file for model management
* `new.py`: Python file for new document processing features
* `podcast.py`: Python file for podcast generation
* `prompts.txt`: File containing chat prompts
* `Readme.md`: This README file
* `requirements.txt`: File listing project dependencies
* `voice_config.py`: Python file for voice configuration

# Setup Instructions

1. Clone the project repository to your local machine.
2. Install the required dependencies listed in `requirements.txt` using pip: `pip install -r requirements.txt`
3. Set up a virtual environment using `venv` or a similar tool.
4. Activate the virtual environment.
5. Run the application using `python main.py`
6. Follow the prompts to upload documents, chat with the system, or create podcast transcripts.

# Features

* Document upload and processing for various file types
* Chat functionality using uploaded data
* Podcast transcript creation
* Support for multiple voice configurations

# Technologies Used

* Python
* OpenAI API
* LangChain
* Google Generative AI Embeddings
* AsyncWebCrawler
* MoviePy
* PyPDFLoader
* CSVLoader
* UnstructuredExcelLoader
* RecursiveCharacterTextSplitter
* Chroma vector store

# Contributing

Contributions to this project are welcome. Please submit pull requests or report issues through the project's issue tracker.

