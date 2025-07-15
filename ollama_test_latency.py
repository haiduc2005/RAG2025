import time
from langchain_ollama import OllamaLLM

# このテストスクリプトは、以下を明確に比較できます。
# ・モデルの初回ロード（コールドスタート）時間
# ・2回目の呼び出し（ウォームアップ後）時間
# ・複数回の呼び出し時間

# Ollamaモデル設定
LLM_MODEL = "deepseek-r1:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollamaが実行中であることを確認してください

# LLMの初期化（注意：このステップではモデルはロードされません）
llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.7,
    base_url=OLLAMA_BASE_URL,
    max_tokens=50
)

# コールドスタート：初回呼び出し
print("🧊 Cold start...")
start = time.time()
response = llm.invoke("你好，请简单介绍一下你自己")
print(f"响应内容: {response}")
print(f"⏱️ 冷启动耗时: {time.time() - start:.2f} 秒")

# しばらく一時停止（モデルが完全にロードされていることを確認）
time.sleep(3)

# ウォームスタート：再呼び出し
print("\n🔥 Warm start...")
start = time.time()
response = llm.invoke("你是谁？")
print(f"响应内容: {response}")
print(f"⏱️ 热启动耗时: {time.time() - start:.2f} 秒")

# 複数回呼び出しのシミュレーション
print("\n🔁 多轮测试...")
questions = ["你能做什么？", "你支持哪些语言？", "你知道什么是RAG吗？"]
for q in questions:
    start = time.time()
    r = llm.invoke(q)
    print(f"[{q}] -> {time.time() - start:.2f} 秒")