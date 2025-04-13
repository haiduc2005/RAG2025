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
print(f"GPU可用: {torch.cuda.is_available()}")  # 应输出True
print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 应显示你的显卡型号
import time
try:
    subprocess.run(["ollama", "ps"], check=True, capture_output=True)
except subprocess.CalledProcessError:
    print("Starting Ollama service...")
    subprocess.Popen(["ollama", "serve"])
    time.sleep(5)

# --- Configuration ---
PDF_DIRECTORY = "data"
PERSIST_DIRECTORY = "./vector_db"
OLLAMA_BASE_URL = "http://localhost:11434" # Adjust if Ollama runs elsewhere
LLM_MODEL = "deepseek-r1:1.5b"
RETRIEVER_K = 3 # Number of relevant chunks to retrieve
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SESSION_ID = "my_rag_session" # Simple session ID for this example

# --- LLM and Embeddings ---
print(f"Initializing LLM: {LLM_MODEL}")

llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.7,
    num_gpu=1,
    base_url=OLLAMA_BASE_URL,
    # format="fp16",
    max_tokens=50,
    num_predict=100,
    num_ctx=2048,
    num_thread=4
    # device="cuda" # This parameter might be ignored; GPU usage is typically configured in Ollama itself.
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",  # 1.3B参数
    model_kwargs={"device": "cuda"}
)
print(embeddings)
# shutil.rmtree(PERSIST_DIRECTORY)

# --- 模型冷启动测试 ---
# start = time.time()
# llm.invoke("warmup")
# print(f"LLM load time: {time.time() - start:.2f} seconds")
# start = time.time()
# embeddings.embed_query("warmup")
# print(f"Embedding load time: {time.time() - start:.2f} seconds")
# --- 模型冷启动 ---
print("Preloading LLM...")
start = time.time()
llm.invoke("")  # 空输入，最小化生成
print(f"LLM preload time: {time.time() - start:.2f} seconds")

# --- Vector Store Setup ---
# vectorstore = None
# 从本地 Chroma 加载向量数据库
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

# --- Retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
print(f"Retriever set up to fetch {RETRIEVER_K} chunks.")

# --- Prompt Template ---
# Updated prompt to include context and instruct the LLM
prompt = ChatPromptTemplate.from_messages(
    [
        # ("system", "あなたは、親切で優秀なアシスタントです。\n\nコンテキスト:\n{context}"),
        # ("system", "あなたは親切で優秀なアシスタントです。\n\nコンテキスト: {context}\n\n過去の会話は参考情報としてのみ使用し、回答には過去の質問や回答を含めず、現在の質問に直接的かつ簡潔に答えてください。"),
        ("system", "你是一个亲切且优秀的助手。\n\n上下文: {context}\n\n请仅将过去的对话作为参考信息，回答时不要包含过去的提问或回答，直接且简洁地回答当前问题。"),
        MessagesPlaceholder(variable_name="chat_history"), # Placeholder for chat history
        ("human", "{human_input}"), # Placeholder for the user's current input
    ]
)

# --- Helper Function to Format Documents ---
def format_docs(docs):
    """Combines document page content into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- RAG Chain Construction ---
# This chain takes the user input, retrieves relevant documents, formats them,
# inserts them into the prompt along with history, calls the LLM, and parses the output.
rag_chain = (
    RunnablePassthrough.assign(
        # Retrieve documents based on "human_input" and format them
        context=lambda x: format_docs(retriever.invoke(x["human_input"]))
    )
    | prompt
    | llm
    | StrOutputParser()
)

# --- History Management ---
# Simple in-memory store for chat histories
store = {}

def get_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates chat history for a given session ID."""
    if session_id not in store:
        print(f"Creating new chat history for session: {session_id}")
        store[session_id] = ChatMessageHistory()
    else:
        if len(store[session_id].messages) > 4:
            store[session_id].messages = store[session_id].messages[-4:]
        print(f"Using existing chat history for session: {session_id}")
    return store[session_id]

# --- Chain with History ---
# Wrap the RAG chain with history management
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="human_input", # The key for the user's input in the input dictionary
    history_messages_key="chat_history", # The key for the history placeholder in the prompt
)
print("\n--- RAG System Ready ---")
# print(f"Using session ID: {SESSION_ID}")
print("Enter your questions. Type 'quit' or 'exit' to stop.")

# --- Interaction Loop ---
while True:
    try:
        human_input = input("\nYou: ")
        if human_input.lower() in ["quit", "exit"]:
            print("Exiting...")
            break
        if not human_input:
            continue

        # Invoke the chain with history
        # The config dictionary passes the session_id for history management
        config = {"configurable": {"session_id": SESSION_ID}}
        # response = chain_with_history.invoke({"human_input": human_input}, config=config)

        # ✅ 记录推理时间
        start_time = time.time()
        response = chain_with_history.invoke({"human_input": human_input}, config=config)
        duration = time.time() - start_time

        print(f"\nAssistant: {response}")
        print(f"(⏱️ Response time: {duration:.2f} seconds)")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Consider adding more robust error handling or logging
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt...")
        break

print("\n--- Chat Session Ended ---")
# Optional: See the final history state
# print("Final History Store:", store)