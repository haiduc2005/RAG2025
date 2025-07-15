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

# --- 設定 ---
PDF_DIRECTORY = "data"
PERSIST_DIRECTORY = "./vector_db"
OLLAMA_BASE_URL = "http://localhost:11434" # Ollamaが別の場所で実行されている場合は調整してください
LLM_MODEL = "deepseek-r1:1.5b"
# EMBEDDING_MODEL = "llama2" # または「nomic-embed-text」、「mxbai-embed-large」など。プルされていることを確認してください
RETRIEVER_K = 3 # 取得する関連チャンクの数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SESSION_ID = "my_rag_session" # この例のシンプルなセッションID
# sentences = ["This is an example sentence", "Each sentence is converted"]

# --- LLMと埋め込み ---
print(f"Initializing LLM: {LLM_MODEL}")
# 注：OllamaLLMは通常、Ollamaサーバーが実行されているデバイスで推論を実行します。
# 「device」パラメータは、ローカルトランスフォーマーのように配置を直接制御しない場合があります。
# 必要に応じて、OllamaサーバーがGPU用に構成されていることを確認してください。
llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.7,
    base_url=OLLAMA_BASE_URL
    # device="cuda" # このパラメータは無視される場合があります。GPUの使用は通常Ollama自体で設定されます。
)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector = embeddings.embed_query("如何使用 DeepSeek 进行 RAG？")
print(vector[:5])  # 出力が長くなりすぎないように、最初の5次元のみを表示
# try:
#         shutil.rmtree(PERSIST_DIRECTORY) # <-- ディレクトリを削除する正しい方法
#         print(f"Successfully deleted directory: {PERSIST_DIRECTORY}")
# except PermissionError as e:
#     print(f"PermissionError deleting directory: {e}")
#     print("./vector_db内のファイルを使用しているプログラム（ファイルエクスプローラーや以前のスクリプト実行など）がないことを確認してください。")
#     print("他のアプリケーションを閉じるか、手動でディレクトリを削除する必要がある場合があります。")
#     exit()
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
print(embeddings)
# ローカルのChromaからベクトルデータベースをロード
vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# vectorstoreに埋め込みがあることを確認
if vectorstore._embedding_function is None:
    raise ValueError("Chroma vector store was loaded without embeddings!")

# --- ベクトルストアのセットアップ ---
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

# --- リトリーバー ---
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
print(f"Retriever set up to fetch {RETRIEVER_K} chunks.")

# --- Prompt Template ---
# コンテキストを含み、LLMに指示するよう更新されたプロンプト
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは親切で優秀なアシスタントです。\n\nコンテキスト:\n{context}"),
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
print(f"Using session ID: {SESSION_ID}")
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
        response = chain_with_history.invoke({"human_input": human_input}, config=config)

        print(f"\nAssistant: {response}")

        # オプション：デバッグのために現在の履歴を印刷
        # print("\n--- Current History ---")
        # print(store[SESSION_ID].messages)
        # print("----------------------")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # より堅牢なエラー処理またはロギングの追加を検討してください
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt...")
        break

print("\n--- Chat Session Ended ---")
# オプション：最終履歴状態を確認
# print("Final History Store:", store)