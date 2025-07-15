import glob
import shutil
import time
import os
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
import subprocess
import torch
print(f"GPU可用: {torch.cuda.is_available()}")  # Trueと表示されるはずです
print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # お使いのグラフィックカードのモデルが表示されるはずです
import time
try:
    subprocess.run(["ollama", "ps"], check=True, capture_output=True)
except subprocess.CalledProcessError:
    print("Ollamaサービスを開始しています... ")
    subprocess.Popen(["ollama", "serve"])
    time.sleep(5)

# --- 設定 ---
PDF_DIRECTORY = "data"
PERSIST_DIRECTORY = "./vector_db"
OLLAMA_BASE_URL = "http://localhost:11434" # Ollamaが別の場所で実行されている場合は調整してください
LLM_MODEL = "deepseek-r1:1.5b"
RETRIEVER_K = 3 # 取得する関連チャンクの数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SESSION_ID = "my_rag_session" # この例のシンプルなセッションID

# --- LLMと埋め込み ---
print(f"Initializing LLM: {LLM_MODEL}")

llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.7,
    num_gpu=1,
    base_url=OLLAMA_BASE_URL,
    # format="fp16",
    max_tokens=100,
    num_predict=100,
    num_ctx=2048,
    num_thread=4
    # device="cuda" # このパラメータは無視される場合があります。GPUの使用は通常Ollama自体で設定されます。
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",  # 1.3Bパラメータ
    model_kwargs={"device": "cuda"}
)
print(embeddings)
# shutil.rmtree(PERSIST_DIRECTORY)

# --- モデルコールドスタートテスト ---
# start = time.time()
# llm.invoke("warmup")
# print(f"LLM load time: {time.time() - start:.2f} seconds")
# start = time.time()
# embeddings.embed_query("warmup")
# print(f"Embedding load time: {time.time() - start:.2f} seconds")
# --- モデルコールドスタート ---
print("LLMをプリロードしています... ")
start = time.time()
llm.invoke("")  # 空の入力、最小限の生成
print(f"LLM preload time: {time.time() - start:.2f} seconds")

# --- ベクトルストアのセットアップ ---
# vectorstore = None
# ローカルのChromaからベクトルデータベースをロード
vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
if not vectorstore._collection.count():

    pdf_files = glob.glob(os.path.join(PDF_DIRECTORY, "*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in '{PDF_DIRECTORY}'.")
        exit()

    docs = []
    for pdf_file in pdf_files:
        try:
            print(f"Loading {pdf_file}...")
            loader = PyPDFLoader(pdf_file)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            continue # Skip problematic files

    print(f"Loaded {len(docs)} document pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents = text_splitter.split_documents(docs)
    print(f"Split documents into {len(documents)} chunks.")

    print("Creating embeddings and vector store...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

# --- リトリーバー ---
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
print(f"Retriever set up to fetch {RETRIEVER_K} chunks.")

# --- プロンプトテンプレート ---
# コンテキストを含み、LLMに指示するよう更新されたプロンプト
prompt = ChatPromptTemplate.from_messages(
    [
        # ("system", "あなたは、親切で優秀なアシスタントです。\n\nコンテキスト:\n{context}"),
        # ("system", "あなたは親切で優秀なアシスタントです。\n\nコンテキスト: {context}\n\n過去の会話は参考情報としてのみ使用し、回答には過去の質問や回答を含めず、現在の質問に直接的かつ簡潔に答えてください。"),
        ("system", "あなたは親切で優秀なアシスタントです。\n\nコンテキスト: {context}\n\n過去の会話は参考情報としてのみ使用し、回答には過去の質問や回答を含めず、現在の質問に直接的かつ簡潔に答えてください。"),
        MessagesPlaceholder(variable_name="chat_history"), # チャット履歴のプレースホルダー
        ("human", "{human_input}"), # ユーザーの現在の入力のプレースホルダー
    ]
)

# --- ドキュメントをフォーマットするためのヘルパー関数 ---
def format_docs(docs):
    """Combines document page content into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- RAGチェーンの構築 ---
# このチェーンは、ユーザー入力を受け取り、関連するドキュメントを取得し、それらをフォーマットし、
# 履歴とともにプロンプトに挿入し、LLMを呼び出し、出力を解析します。
rag_chain = (
    RunnablePassthrough.assign(
        # 「human_input」に基づいてドキュメントを取得し、それらをフォーマットします
        context=lambda x: format_docs(retriever.invoke(x["human_input"]))
    )
    | prompt
    | llm
    | StrOutputParser()
)

# --- 履歴管理 ---
# チャット履歴用のシンプルなインメモリストア
store = {}

def get_history(session_id: str) -> BaseChatMessageHistory:
    """指定されたセッションIDのチャット履歴を取得または作成します。"""
    if session_id not in store:
        print(f"Creating new chat history for session: {session_id}")
        store[session_id] = ChatMessageHistory()
    else:
        if len(store[session_id].messages) > 4:
            store[session_id].messages = store[session_id].messages[-4:]
        print(f"Using existing chat history for session: {session_id}")
    return store[session_id]

# --- 履歴付きチェーン ---
# RAGチェーンを履歴管理でラップします
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="human_input", # 入力辞書内のユーザー入力のキー
    history_messages_key="chat_history", # プロンプト内の履歴プレースホルダーのキー
)
print("\n--- RAG System Ready ---")
# print(f"Using session ID: {SESSION_ID}")
print("Enter your questions. Type 'quit' or 'exit' to stop.")

# --- 対話ループ ---
while True:
    try:
        human_input = input("\nYou: ")
        if human_input.lower() in ["quit", "exit"]:
            print("Exiting...")
            break
        if not human_input:
            continue

        # Invoke the chain with history
        # config辞書は履歴管理のためにsession_idを渡します
        config = {"configurable": {"session_id": SESSION_ID}}
        # response = chain_with_history.invoke({"human_input": human_input}, config=config)

        # ✅ 推論時間を記録
        start_time = time.time()
        response = chain_with_history.invoke({"human_input": human_input}, config=config)
        duration = time.time() - start_time

        print(f"\nAssistant: {response}")
        print(f"(⏱️ Response time: {duration:.2f} seconds)")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # より堅牢なエラー処理またはロギングの追加を検討してください
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt...")
        break

print("\n--- Chat Session Ended ---")
# オプション：最終履歴状態を確認
# print("Final History Store:", store)