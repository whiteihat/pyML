"""
项目路径工具模块
"""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def project_root(mark="pyproject.toml") -> Path:
    """
    从当前文件位置开始，向上查找包含标记文件的根目录。

    Args:
        mark (str): 用作标记的文件名，默认为 'pyproject.toml'。

    Returns:
        Path: 项目根目录的 Path 对象。
    """

    # 从当前文件 (utils.py) 所在的目录开始向上查找, 直到找到标记文件或到达文件系统根目录
    dir = Path(__file__).resolve().parent
    while dir != dir.parent:
        if (dir / mark).exists():
            return dir
        dir = dir.parent

    raise FileNotFoundError(f"未找到包含 '{mark}' 的项目根目录。")
