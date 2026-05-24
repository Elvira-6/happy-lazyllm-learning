import lazyllm
import os

llm = lazyllm.OnlineChatModule(
    source="qwen",	# 指定模型来源是通义千问
    model="qwen-plus-latest",  # 通义千问的模型名
    api_key=os.getenv("QWEN_API_KEY"),
)

lazyllm.WebModule(llm, port=23466, history=[llm]).start().wait()
