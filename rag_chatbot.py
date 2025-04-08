import glob
import shutil
import time
import os
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
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
print(f"GPU可用: {torch.cuda.is_available()}")  # 应输出True
print(f"GPU名称: {torch.cuda.get_device_name(0)}")  # 应显示你的显卡型号


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
    max_tokens=50
    # device="cuda" # This parameter might be ignored; GPU usage is typically configured in Ollama itself.
)
start = time.time()
response = llm.invoke("llm测试")
print(f"Inference time: {time.time() - start:.2f} seconds")
# embeddings = OllamaEmbeddings(
#     model="nomic-embed-text",
#     num_gpu=1
#     # num_thread=4,  # 使用部分CPU分担
#     # batch_size=8   # 减小批处理大小
# )

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-zh-v1.5",  # 1.3B参数
    model_kwargs={"device": "cuda"}
)
text = "这是一个测试句子" * 100  # 模拟长文本
start = time.time()
embedding = embeddings.embed_query(text)
print(f"Embedding time: {time.time() - start:.2f} seconds")
# vector = embeddings.embed_query("如何使用 DeepSeek 进行 RAG？")
# print(vector[:5])  # 只显示前 5 维，避免输出太长
print(embeddings)
# shutil.rmtree(PERSIST_DIRECTORY)

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
start = time.time()
docs = vectorstore.similarity_search("vectorstore测试", k=3)
print(f"Retrieval time: {time.time() - start:.2f} seconds")
# --- Retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
print(f"Retriever set up to fetch {RETRIEVER_K} chunks.")

# --- Prompt Template ---
# Updated prompt to include context and instruct the LLM
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは、親切で優秀なアシスタントです。\n\nコンテキスト:\n{context}"),
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
# start = time.time()
# response = chain_with_history.invoke({"question": "rag_chain测试"}, config={"configurable": {"session_id": "default"}})
# print(f"Total time: {time.time() - start:.2f} seconds")
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
        response = chain_with_history.invoke({"human_input": human_input}, config=config)

        print(f"\nAssistant: {response}")

        # Optional: Print current history for debugging
        # print("\n--- Current History ---")
        # print(store[SESSION_ID].messages)
        # print("----------------------")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Consider adding more robust error handling or logging
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt...")
        break

print("\n--- Chat Session Ended ---")
# Optional: See the final history state
# print("Final History Store:", store)