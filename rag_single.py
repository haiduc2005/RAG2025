import glob
import shutil
import os
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
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
    device="cuda" # This parameter might be ignored; GPU usage is typically configured in Ollama itself.
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    num_gpu=1
    # num_thread=4,  # 使用部分CPU分担
    # batch_size=8   # 减小批处理大小
)
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

# --- Retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
print(f"Retriever set up to fetch {RETRIEVER_K} chunks.")

# 定义提示模板
prompt_template = """
基于以下上下文回答问题。如果无法回答，说“我不知道”。
上下文: {context}
问题: {question}
回答:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# 创建 RAG 链
rag_chain = (
    {"context": retriever | (lambda docs: "\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 测试
question = "如何使用在本地利用DeepSeek？"
response = rag_chain.invoke(question)
print(f"问题: {question}")
print(f"回答: {response}")