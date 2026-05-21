import os
import hashlib
import json
from typing import Optional


class DedupManager:
    """文档去重管理：基于 MD5 哈希判断文件是否已入库"""

    def __init__(self, record_path: str = "./data/md5_records.json"):
        self.record_path = record_path
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        self.records = self._load_records()
        self._filename_index = self._build_filename_index()

    # ── 内部方法 ──────────────────────────

    def _load_records(self) -> dict:
        """加载已有的 MD5 记录"""
        if os.path.exists(self.record_path):
            with open(self.record_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_records(self):
        """持久化 MD5 记录"""
        with open(self.record_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def _build_filename_index(self) -> dict:
        """从 records 重建 filename → [md5, ...] 映射"""
        idx = {}
        for md5, info in self.records.items():
            fname = info.get("filename")
            if fname:
                idx.setdefault(fname, []).append(md5)
        return idx

    def _update_filename_index(self, filename: str, md5: str, remove: bool = False):
        """同步更新文件名索引"""
        if remove:
            if filename in self._filename_index:
                self._filename_index[filename] = [
                    h for h in self._filename_index[filename] if h != md5
                ]
                if not self._filename_index[filename]:
                    del self._filename_index[filename]
        else:
            self._filename_index.setdefault(filename, []).append(md5)

    # ── 静态工具 ──────────────────────────

    @staticmethod
    def compute_md5(file_path: str) -> str:
        """计算文件的 MD5 哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    # ── 去重查询 ──────────────────────────

    def is_duplicated(self, md5_hash: str) -> bool:
        """检查 MD5 是否已存在（即是否已入库过）"""
        return md5_hash in self.records

    # ── 增删操作 ──────────────────────────

    def add_record(self, file_path: str, md5_hash: Optional[str] = None):
        """添加一条入库记录（以 md5_hash 为 key）"""
        if md5_hash is None:
            md5_hash = self.compute_md5(file_path)

        filename = os.path.basename(file_path)
        self.records[md5_hash] = {
            "filename": filename,
            "file_path": file_path,
            "md5": md5_hash,
        }
        self._update_filename_index(filename, md5_hash, remove=False)
        self._save_records()

    def remove_record(self, md5_hash: str):
        """删除一条记录（通过 md5 哈希）"""
        if md5_hash in self.records:
            info = self.records[md5_hash]
            self._update_filename_index(
                info.get("filename", ""), md5_hash, remove=True
            )
            del self.records[md5_hash]
            self._save_records()

    def remove_by_filename(self, filename: str) -> bool:
        """用户友好删除：通过文件名删除"""
        md5_list = self._filename_index.get(filename, [])
        if not md5_list:
            return False
        for md5 in md5_list[:]:
            self.remove_record(md5)
        return True

    # ── 查询 ──────────────────────────────

    def get_all_records(self) -> dict:
        """获取所有已入库的文件记录"""
        return self.records

    def get_record_by_filename(self, filename: str) -> list[dict]:
        """通过文件名查询记录"""
        md5_list = self._filename_index.get(filename, [])
        return [self.records[md5] for md5 in md5_list if md5 in self.records]


def add_with_dedup(file_path: str, dedup_mgr: DedupManager) -> bool:
    """
    带去重检查的文件添加
    返回 True 表示需要入库（新文件），False 表示已存在（跳过）
    """
    md5_hash = dedup_mgr.compute_md5(file_path)
    if dedup_mgr.is_duplicated(md5_hash):
        print(f"⏭️  跳过已入库的文件：{os.path.basename(file_path)}")
        return False
    return True
