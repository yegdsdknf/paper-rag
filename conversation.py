"""多轮对话管理器：存储历史 + 改写追问"""
import re

from utils.ollama_client import create_chat_ollama


SOURCE_NAMES = ["BERT", "GPT-3", "GPT3", "T5", "ViT", "DeepSeek-R1", "DeepSeek-R1-Zero", "FFA-Net"]


def _source_from_history(history: list[dict]) -> str | None:
    text = "\n".join(str(item.get("content", "")) for item in history[-6:])
    for source in SOURCE_NAMES:
        if re.search(re.escape(source), text, flags=re.IGNORECASE):
            return "GPT-3" if source.upper() == "GPT3" else source
    return None


def _needs_source_followup(question: str) -> bool:
    q_lower = question.lower()
    return any(signal in q_lower for signal in ["另一个", "这个", "它", "another", "the other"])


def _ensure_followup_source(question: str, rewritten: str, history: list[dict]) -> str:
    if not _needs_source_followup(question):
        return rewritten
    source = _source_from_history(history)
    if not source or re.search(re.escape(source), rewritten, flags=re.IGNORECASE):
        return rewritten
    return f"{source} {rewritten}"


class ConversationManager:
    def __init__(
        self,
        llm_model: str,
        temperature: float,
        num_ctx: int | None = None,
        num_predict: int | None = None,
    ):
        self.llm_model = llm_model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.llm = create_chat_ollama(
            model=llm_model,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
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
        self.llm = create_chat_ollama(
            model=llm_model,
            temperature=temperature,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )
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
        return _ensure_followup_source(question, result.strip(), self.history)

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
