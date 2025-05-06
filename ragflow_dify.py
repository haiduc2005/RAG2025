import glob
import shutil
import time
import os
import pandas as pd
import chardet
from typing import List
from fastapi import FastAPI, UploadFile, File
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import torch

# --- Configuration ---
EXL_DIRECTORY = "data/excel"
PERSIST_DIRECTORY = "./vector_db"
OLLAMA_BASE_URL = "http://localhost:11434" # Adjust if Ollama runs elsewhere
LLM_MODEL = "deepseek-r1:1.5b"
RETRIEVER_K = 3 # Number of relevant chunks to retrieve
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SESSION_ID = "my_rag_session" # Simple session ID for this example

app = FastAPI()

def detect_encoding(file_bytes: bytes) -> str:
    """自动检测文件编码"""
    result = chardet.detect(file_bytes[:10000])  # 检测前10KB
    return result['encoding'] or 'utf-8'


@app.post("/upload-multi-csv/")
async def upload_csv(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        # 读取文件内容为字节
        content = await file.read()

        # 检测编码（处理日文）
        encoding = detect_encoding(content)

        try:
            # 转换为DataFrame（处理日文列名和数据）
            df = pd.read_csv(
                io.StringIO(content.decode(encoding)),
                encoding=encoding,
                engine='python'  # 更兼容日文
            )

            # 替换列名中的不可见字符
            df.columns = df.columns.str.strip()

            results.append({
                "filename": file.filename,
                "status": "success",
                "columns": list(df.columns),
                "sample_data": df.head(2).to_dict(orient="records")
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })

    return {"files": results}