"""Prompt 加载器 — 从文件读模板，改提示词不用改代码"""

import os


def load_prompt(prompt_name: str, prompts_dir: str = "./prompts") -> str:
    """
    加载指定名称的 prompt 模板

    用法：
        from prompt_loader import load_prompt
        template = load_prompt("rag_summary_prompt")
    """
    file_path = os.path.join(prompts_dir, f"{prompt_name}.txt")
    if not os.path.exists(file_path):
        print(f"❌ Prompt 文件不存在：{file_path}")
        return ""

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"✅ Prompt 已加载：{file_path}")
    return content
