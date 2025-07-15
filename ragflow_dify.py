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

# --- 設定 ---
EXL_DIRECTORY = "data/excel"
PERSIST_DIRECTORY = "./vector_db"
OLLAMA_BASE_URL = "http://localhost:11434" # Ollamaが別の場所で実行されている場合は調整してください
LLM_MODEL = "deepseek-r1:1.5b"
RETRIEVER_K = 3 # 取得する関連チャンクの数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SESSION_ID = "my_rag_session" # この例のシンプルなセッションID

app = FastAPI()

def detect_encoding(file_bytes: bytes) -> str:
    """ファイルのエンコーディングを自動検出"""
    result = chardet.detect(file_bytes[:10000])  # 最初の10KBを検出
    return result['encoding'] or 'utf-8'


@app.post("/upload-multi-csv/")
async def upload_csv(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        # ファイルの内容をバイトとして読み込む
        content = await file.read()

        # エンコーディングを検出（日本語を処理）
        encoding = detect_encoding(content)

        try:
            # DataFrameに変換（日本語の列名とデータを処理）
            df = pd.read_csv(
                io.StringIO(content.decode(encoding)),
                encoding=encoding,
                engine='python'  # 日本語との互換性が向上
            )

            # 列名内の不可視文字を置換
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