"""
交互式问答入口
提供命令行交互界面，支持连续对话
"""

from rag_pipeline import build_hybrid_retriever, route_question, ask_with_context
from conversation import ConversationManager
from utils.config_loader import load_config
from utils.console import ensure_utf8_console

config = load_config()

ensure_utf8_console()

def print_divider():
    print("\n" + "-" * 60)


def main():
    print("=" * 60)
    print("📚  论文知识库问答系统".center(58))
    print("=" * 60)
    # 初始化
    print("\n⏳ 正在初始化（约需 5-10 秒）...")
    retriever = build_hybrid_retriever()
    conv = ConversationManager(
        llm_model=config["llm_model"],
        temperature=config["temperature"],
    )

    print("\n✅ 初始化完成！输入 q 退出，输入 s 查看提示, 输入 c 清空对话历史")

    # 交互循环
    while True:
        print_divider()
        question = input("\n💬 请提问：\n> ").strip()
        if question.lower() == "q":
            print("\n👋 再见！")
            break

        if question.lower() == "c":
            conv.clear()
            print("🧹 对话历史已清空")
            continue

        if question.lower() == "s":
            print("""
        📋 使用提示：
          - 直接输入问题即可获得回答
          - 支持追问，上下文保留
          - 输入 c 清空对话历史
          - 输入 q 退出程序
          - 建议问题示例：
            · "这篇论文提出了什么方法？"
            · "实验用了什么数据集？"
            · "论文的主要贡献是什么？"
                    """)
            continue

        if not question:
            continue

        # 执行问答
        print("\n🤔 思考中...")
        try:
            answer, sources = ask_with_context(retriever, conv, question)
            conv.add_turn(question, answer)
            # 打印回答
            print(f"\n📝 回答：\n{answer}")

            # 打印参考来源
            if sources:
                print(f"\n📎 参考来源：")
                for i, doc in enumerate(sources):
                    source_file = doc.metadata.get("source", "未知")
                    page = doc.metadata.get("page", "?")
                    # 只显示文件名，不显示完整路径
                    short_name = source_file.split("/")[-1]
                    print(f"  [{i + 1}] {short_name} 第{page}页")
        except Exception as e:
            print(f"\n❌ 出错了：{e}")
            print("试试重新提问，或检查 Ollama 服务是否正常运行")


if __name__ == '__main__':
    main()
