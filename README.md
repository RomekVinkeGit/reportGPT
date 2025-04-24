# Reshaped case

An implementation of a Retrieval-Augmented Generation (RAG) system and table parsing system using LangChain, Chroma, OpenAI, and PDF processing utilities.

## Setup

1. Clone this repository
2. Install Python 3.11:
   - **Option 1 (Recommended)**: Install via Microsoft Store
     - Open Microsoft Store
     - Search for "Python 3.11"
     - Click "Get" or "Install"
   - **Option 2**: Install via Python website
     - Visit https://www.python.org/downloads/release/python-3116/
     - Download "Windows installer (64-bit)"
     - Run the installer
     - **Important**: Check "Add Python 3.11 to PATH" during installation
   - **Option 3**: Install via winget
     ```bash
     winget install Python.Python.3.11
     ```

3. Create and set up a virtual environment:
   ```bash
   # Create virtual environment with Python 3.11
   py -3.11 -m venv venv-py311

   # Activate the virtual environment
   # On Windows:
   .\venv-py311\Scripts\activate
   # On Unix/Mac:
   source venv-py311/bin/activate

   # Install required packages
   pip install -r requirements.txt

   # Install Jupyter and ipykernel
   pip install jupyter notebook ipykernel

   # Register the virtual environment as a Jupyter kernel
   python -m ipykernel install --user --name=venv-py311 --display-name="Python 3.11 (venv-py311)"
   ```

4. Create a `.env` file in the project root with your OpenAI API credentials:
   ```
   OPENAI_API_VERSION=your_api_version
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   AZURE_OPENAI_API_KEY=your_api_key_here
   ```

5. Using the virtual environment in Jupyter:
   - Start Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - When creating a new notebook, select "Python 3.11 (venv-py311)" from the kernel dropdown
   - Or in an existing notebook, go to "Kernel" → "Change kernel" and select "Python 3.11 (venv-py311)"

## Project Structure

- `utils.py`: Core utility functions for PDF processing, vector database creation, retrieval, and table extraction
- `question_1.ipynb` and `question_2.ipynb`: Main entry points to the code, demonstrating different use cases
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (create this file)

## Usage

The Jupyter notebooks (`question_1.ipynb` and `question_2.ipynb`) are the main entry points to the code. They demonstrate different use cases and workflows:

1. Open either `question_1.ipynb` or `question_2.ipynb` in Jupyter
2. Follow the notebook cells to:
   - Load and process PDF documents
   - Create vector databases for semantic search
   - Query the RAG system with specific questions
   - Extract tables using different methods (LLM-based or Camelot-based)

## Features

- PDF document loading and chunking with configurable parameters
- Vector database creation with Chroma for semantic search
- Question answering using RAG (Retrieval Augmented Generation)
- Table extraction using both LLM-based and Camelot-based approaches

## Dependencies

- OpenAI API (Azure or standard) for embeddings and generation
- PyMuPDF (fitz) for PDF text extraction
- Camelot for table extraction
- Langchain for document processing and vector storage
- Pandas for data manipulation