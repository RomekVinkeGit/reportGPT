"""
PDF Processing and Question Answering Utilities

This module provides a comprehensive set of utilities for processing PDF documents and performing
question answering using a combination of vector storage and language models. It supports various
PDF processing tasks including text extraction, table extraction, and semantic search capabilities.

The module integrates with OpenAI's API (both Azure and standard) and uses multiple PDF processing
libraries (PyMuPDF and Camelot) for different extraction needs.

Main Features:
    - PDF text extraction and chunking with configurable parameters
    - Vector database creation and semantic search using Chroma
    - Table extraction using both LLM-based and Camelot-based approaches
    - Question answering using RAG (Retrieval Augmented Generation)
    - Warning suppression for third-party libraries

Dependencies:
    - OpenAI API key or Azure OpenAI credentials in environment variables
    - PyMuPDF (fitz) for PDF text extraction
    - Camelot for table extraction
    - Langchain for document processing and vector storage
    - Pandas for data manipulation

Environment Variables Required:
    - OPENAI_API_VERSION: Version of the OpenAI API
    - AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
    - AZURE_OPENAI_API_KEY: Azure OpenAI API key
"""


import os
from typing import List
from io import StringIO
import warnings
from contextlib import contextmanager
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
import fitz
import pandas as pd

def load_pdf_with_recursive_splitter(
    file_path: str, chunk_size: int = 500, chunk_overlap: int = 100
) -> List[Document]:
    """
    Load a PDF file and split it into chunks using RecursiveCharacterTextSplitter.

    Args:
        file_path: Path to the PDF file
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of Document objects containing text chunks
    """
    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    # Split the documents into chunks
    chunks = text_splitter.split_documents(pages)

    return chunks


def create_vector_db(chunks: List[Document], persist_directory: str = "db") -> Chroma:
    """
    Create a Chroma vector database from text chunks using Azure OpenAI embeddings.

    Args:
        chunks: List of Document objects
        persist_directory: Directory to store the vector database

    Returns:
        Chroma vector database
    """
    # Use AzureOpenAIEmbeddings to handle embeddings with Azure OpenAI
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        model="text-embedding-ada-002"
    )

    # Creating and returning the Chroma vector database directly from the embeddings
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,  # Pass the embeddings object
        persist_directory=persist_directory
    )

    return vector_db

def retrieve_relevant_chunks(
    vector_db: Chroma, query: str, k: int = 5
) -> List[Document]:
    """
    Retrieve the most relevant chunks using maximum marginal relevance search.
    This approach balances relevance to the query with diversity among the results.

    Args:
        vector_db: Chroma vector database
        query: User query
        k: Number of chunks to retrieve

    Returns:
        List of relevant Document objects
    """
    # Get results using max marginal relevance search
    # lambda_mult=0.5 balances between relevance and diversity
    results = vector_db.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=k * 3,  # Fetch more candidates for better diversity selection
        lambda_mult=0.5,  # Balance between relevance and diversity
    )

    # Combine chunks into context
    context = "\n\n".join([chunk.page_content for chunk in results])

    return context


def setup_prompt(context: str) -> str:
    """
    Set up the prompt for the RAG system by combining the system message with the context.

    Args:
        context: The context to include in the prompt

    Returns:
        str: The complete prompt with system message and context
    """
    
    system_message = """You are a research assistant helping a team prepare for client discovery meetings.

    You will be given context from a company's annual report. Your job is to answer specific questions about the company by using only the provided 
    context. Do not guess or use external knowledge — rely solely on the context you are given.

    Be clear, concise, and objective. If the answer is not present in the context, say: "The information is not available in the provided document."

    Format your answers in complete sentences. Use bullet points if listing key points or risks. Use professional tone, 
    as your response will be reviewed by consultants preparing for client meetings.
    
    The information you are given about the company is the following"""

    # Combine system message with context
    full_prompt = f"{system_message}\n\nContext:\n{context}"

    return full_prompt


def query_llm(system_prompt: str, question: str, temperature: float = 0) -> str:
    """
    Query the LLM with the given prompt and question.

    Args:
        prompt: The complete prompt with system message and context
        question: The question to answer
        temperature: The temperature setting for the LLM (default: 0)

    Returns:
        str: The LLM's response
    """
    # Setting up the Azure OpenAI client with required credentials and endpoint
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION")
    )

    # Set up the messages for the chat completion
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # Request a completion from the OpenAI Chat API
    completion = client.chat.completions.create(
        model="gpt-35-turbo", messages=messages, temperature=temperature
    )

    # Return the response content
    return completion.choices[0].message.content


def extract_last_page_text(pdf_path: str) -> str:
    """
    Extracts text content from the last page of a PDF file.

    Args:
        pdf_path (str): Path to the PDF file to process

    Returns:
        str: Raw text content extracted from the last page
    """
    doc = fitz.open(pdf_path)
    last_page = doc[-1]  # last page
    text = last_page.get_text()

    return text


def extract_tables_with_llm(text: str, temperature: float = 0) -> pd.DataFrame:
    """
    Converts raw text into a structured DataFrame using GPT model.

    This function:
    1. Initializes OpenAI client
    2. Sends text to GPT with formatting instructions
    3. Converts the response into a pandas DataFrame

    Args:
        text (str): Raw text content to be converted into a table

    Returns:
        pd.DataFrame: Structured table data from the text
    """
    # Define prompt with specific formatting instructions
    system_prompt = f"""The following is a raw table extracted from a PDF. Convert it into a clean CSV format following these rules:
    1. Each row should have the same number of columns
    2. Use semicolons to separate columns
    3. Preserve the header row
    4. Remove any empty rows
    5. Preserve the row structure of the table
    6. Remove any extra whitespace
    7. Do not add any explanatory text, just return the CSV data

    Raw text:
    {text}"""

    # Setting up the Azure OpenAI client with required credentials and endpoint
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )

    # Set up the messages for the chat completion
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Return the table"},
    ]

    # Request a completion from the OpenAI Chat API
    completion = client.chat.completions.create(
        model="gpt-35-turbo", messages=messages, temperature=temperature
    )

    # Convert response to DataFrame
    csv_text = completion.choices[0].message.content.strip()
    df = pd.read_csv(StringIO(csv_text), sep=";")

    return df


@contextmanager
def suppress_camelot_warnings():
    """
    Context manager to temporarily suppress Camelot warnings and logging.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Suppress logging
    logging.getLogger("camelot").setLevel(logging.ERROR)

    try:
        yield
    finally:
        # Reset warnings
        warnings.resetwarnings()

        # Reset logging
        logging.getLogger("camelot").setLevel(logging.INFO)


def extract_tables_with_camelot(pdf_path: str, min_columns: int = 4) -> list:
    """
    Extracts tables from a PDF file using Camelot and filters them based on column count.

    This function uses Camelot's stream parser to extract tables from a PDF file and
    filters them to include only tables with more than the specified number of columns.

    Args:
        pdf_path (str): Path to the PDF file to process
        min_columns (int, optional): Minimum number of columns required to include a table.
            Defaults to 4.

    Returns:
        list: List of pandas DataFrames, each containing a filtered table
    """
    import camelot as camelot

    with suppress_camelot_warnings():
        # Extract all tables using Camelot's stream parser
        stream_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')

        # Filter tables based on column count
        filter_tables = []
        for table in stream_tables:
            table = table.df
            if len(table.columns) > min_columns:
                filter_tables.append(table)

    return filter_tables
