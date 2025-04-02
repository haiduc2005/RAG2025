import glob
import shutil
import os
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
# from sentence_transformers import SentenceTransformer

# --- Configuration ---
PDF_DIRECTORY = "data"
PERSIST_DIRECTORY = "./vector_db"
OLLAMA_BASE_URL = "http://localhost:11434" # Adjust if Ollama runs elsewhere
LLM_MODEL = "deepseek-r1:1.5b"
# EMBEDDING_MODEL = "llama2" # Or "nomic-embed-text", "mxbai-embed-large", etc. Ensure it's pulled
RETRIEVER_K = 3 # Number of relevant chunks to retrieve
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SESSION_ID = "my_rag_session" # Simple session ID for this example
# sentences = ["This is an example sentence", "Each sentence is converted"]

# --- LLM and Embeddings ---
print(f"Initializing LLM: {LLM_MODEL}")
# Note: OllamaLLM usually runs inference on the device where the Ollama server is running.
# The 'device' parameter might not directly control placement like in local transformers.
# Ensure your Ollama server is configured for GPU if desired.
llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.7,
    base_url=OLLAMA_BASE_URL
    # device="cuda" # This parameter might be ignored; GPU usage is typically configured in Ollama itself.
)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector = embeddings.embed_query("如何使用 DeepSeek 进行 RAG？")
print(vector[:5])  # 只显示前 5 维，避免输出太长
# try:
#         shutil.rmtree(PERSIST_DIRECTORY) # <-- CORRECT WAY TO DELETE DIRECTORY
#         print(f"Successfully deleted directory: {PERSIST_DIRECTORY}")
# except PermissionError as e:
#     print(f"PermissionError deleting directory: {e}")
#     print("Please ensure no programs (like file explorer or previous script runs) are using files inside ./vector_db.")
#     print("You might need to close other applications or manually delete the directory.")
#     exit()
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
print(embeddings)
# 从本地 Chroma 加载向量数据库
vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# 确保 vectorstore 具有 embeddings
if vectorstore._embedding_function is None:
    raise ValueError("Chroma vector store was loaded without embeddings!")

# --- Vector Store Setup ---
# vectorstore = None
# if os.path.exists(PERSIST_DIRECTORY):
#     print(f"Loading existing vector store from {PERSIST_DIRECTORY}")
#     vectorstore = Chroma(
#         persist_directory=PERSIST_DIRECTORY,
#         embedding_function=embeddings
#     )
# else:
print(f"Creating new vector store from PDFs in {PDF_DIRECTORY}")
if not os.path.exists(PDF_DIRECTORY):
    print(f"Error: PDF directory '{PDF_DIRECTORY}' not found.")
    exit()

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

if not docs:
    print("Error: No documents were successfully loaded.")
    exit()

print(f"Loaded {len(docs)} document pages.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
documents = text_splitter.split_documents(docs)
print(f"Split documents into {len(documents)} chunks.")

if not documents:
    print("Error: No chunks were created after splitting.")
    exit()

print("Creating embeddings and vector store...")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY
)
    # print(f"Vector store created and persisted at {PERSIST_DIRECTORY}")

if vectorstore is None:
    print("Error: Vector store initialization failed.")
    exit()

# --- Retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
print(f"Retriever set up to fetch {RETRIEVER_K} chunks.")

# --- Prompt Template ---
# Updated prompt to include context and instruct the LLM
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは親切で優秀なアシスタントです。\n\nコンテキスト:\n{context}"),
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

print("\n--- RAG System Ready ---")
print(f"Using session ID: {SESSION_ID}")
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