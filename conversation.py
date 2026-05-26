"""多轮对话管理器：存储历史 + 改写追问"""
from langchain_ollama import ChatOllama


class ConversationManager:
    def __init__(self, llm_model: str, temperature: float):
        self.llm_model = llm_model
        self.temperature = temperature
        self.llm = ChatOllama(model=llm_model, temperature=temperature)
        self.history: list[dict] = []  # [{"role":"user","content":"..."}, ...]

        # 启动时验证 LLM 连通性
        try:
            self.llm.invoke("ping")
            print(f"✅ 对话管理器就绪（{llm_model}）")
        except Exception as e:
            print(f"⚠️  对话管理器 LLM 连接失败：{e}，改写追问功能可能不可用")
            self.llm = None

    def update_model(self, llm_model: str, temperature: float | None = None):
        """切换用于追问改写的模型；保持历史不变。"""
        if temperature is None:
            temperature = self.temperature
        if self.llm_model == llm_model and self.temperature == temperature:
            return
        self.llm_model = llm_model
        self.temperature = temperature
        self.llm = ChatOllama(model=llm_model, temperature=temperature)
        try:
            self.llm.invoke("ping")
            print(f"✅ 对话管理器模型已切换为（{llm_model}）")
        except Exception as e:
            print(f"⚠️  对话管理器模型切换失败：{e}，改写追问功能可能不可用")
            self.llm = None

    def add_turn(self, question: str, answer: str):
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        # 保留最近 6 轮（12 条消息），避免 prompt 过长
        if len(self.history) > 12:
            self.history = self.history[-12:]

    def reformulate(self, question: str) -> str:
        """用对话历史把追问改写成独立问题"""
        if not self.history or self.llm is None:
            return question

        history_text = "\n".join(
            f"{'用户' if h['role'] == 'user' else '助手'}: {h['content'][:300]}"
            for h in self.history[-6:]
        )

        prompt = f"""根据以下对话历史，将用户的最新追问改写为一个独立完整的问题，使问题脱离上下文也能被理解。如果问题本身已经是独立的，原样返回。

        对话历史：
        {history_text}

        追问：{question}

        独立问题："""
        response = self.llm.invoke(prompt)
        result = response.content if hasattr(response, "content") else str(response)
        return result.strip()

    def format_history(self) -> str:
        """格式化历史，用于注入生成 prompt"""
        if not self.history:
            return ""
        lines = ["\n--- 对话历史 ---"]
        for h in self.history[-6:]:
            role = "用户" if h["role"] == "user" else "助手"
            lines.append(f"{role}: {h['content'][:300]}")
        lines.append("--- 当前问题 ---")
        return "\n".join(lines)

    def clear(self):
        self.history = []
