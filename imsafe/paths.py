"""项目根目录与数据文件路径（避免依赖当前工作目录）。"""
import os

# 包所在目录的上一级 = 项目根（含 templates、static、数据库与模型）
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_project_root() -> str:
    return os.path.dirname(_PACKAGE_DIR)


def get_db_path() -> str:
    return os.path.join(get_project_root(), "security_system.db")


def data_path(*parts: str) -> str:
    """相对于项目根的路径。"""
    return os.path.join(get_project_root(), *parts)
