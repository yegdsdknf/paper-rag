"""
论文知识库问答系统 - 统一入口

用法：
  python main.py build     # 论文入库
  python main.py query     # 启动问答
  python main.py doctor    # 启动诊断
  python main.py           # 默认进入问答模式
"""

import sys
import importlib

from utils.console import ensure_utf8_console


ensure_utf8_console()

def main():
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = "query"

    if command == "build":
        print("🏗️  启动论文入库...\n")
        import build_knowledge
        build_knowledge.main(sys.argv[2:])
    elif command == "query":
        print("📚 启动问答系统...\n")
        import query
        query.main()
    elif command == "doctor":
        diagnostics = importlib.import_module("paper_rag.config.diagnostics")
        return diagnostics.run_doctor_cli(sys.argv[2:])
    elif command in ("-h", "--help"):
        print("""用法：
      python main.py build     论文入库（首次运行或新增论文后执行）
      python main.py query     启动交互式问答（默认）
      python main.py doctor    运行启动诊断
      python main.py -h        显示帮助
            """)
    else:
        print(f"❌ 未知命令：{command}")
        print("使用 python main.py -h 查看帮助")


if __name__ == '__main__':
    sys.exit(main() or 0)
