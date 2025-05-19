import time
from langchain_ollama import OllamaLLM

# 测试脚本，能够清晰对比：
# ・模型首次加载（冷启动）时间
# ・第二次调用（预热后）时间
# ・多轮调用时间

# Ollama 模型设置
LLM_MODEL = "deepseek-r1:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"  # 确保 Ollama 正在运行

# 初始化 LLM（注意，这一步不会加载模型）
llm = OllamaLLM(
    model=LLM_MODEL,
    temperature=0.7,
    base_url=OLLAMA_BASE_URL,
    max_tokens=50
)

# 冷启动：首次调用
print("🧊 Cold start...")
start = time.time()
response = llm.invoke("你好，请简单介绍一下你自己")
print(f"响应内容: {response}")
print(f"⏱️ 冷启动耗时: {time.time() - start:.2f} 秒")

# 暂停一会（确保模型已加载完）
time.sleep(3)

# 热启动：再次调用
print("\n🔥 Warm start...")
start = time.time()
response = llm.invoke("你是谁？")
print(f"响应内容: {response}")
print(f"⏱️ 热启动耗时: {time.time() - start:.2f} 秒")

# 多轮调用模拟
print("\n🔁 多轮测试...")
questions = ["你能做什么？", "你支持哪些语言？", "你知道什么是RAG吗？"]
for q in questions:
    start = time.time()
    r = llm.invoke(q)
    print(f"[{q}] -> {time.time() - start:.2f} 秒")