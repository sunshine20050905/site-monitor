"""
桌面端业务常量（与 UI 分离，便于维护与答辩说明）。
监控、数据库路径等业务逻辑仍以 desktop_app 为主入口。
"""

WINDOW_TITLE = "工地人员识别与安全监控"
MAIN_HEADLINE = "工地人员识别与安全监控"
SUBTITLE = "中国大学生计算机设计大赛 · 桌面工作站"

GEOMETRY = "1280x820"
MINSIZE = (1040, 720)

PERMISSION_LEVELS = {
    0: "无权限",
    1: "A级（办公区）",
    2: "B级（车间区）",
    3: "C级（仓库区）",
    4: "D级（施工区）",
}

AREAS = {
    "main_gate": "主入口",
    "office": "办公区",
    "workshop": "车间区",
    "warehouse": "仓库区",
    "construction": "施工区",
}

HELMET_REQUIRED_AREAS = ("workshop", "warehouse", "construction")

VIOLATION_TYPES = {
    "no_helmet": "未戴安全帽",
    "smoking": "抽烟",
    "no_permission": "权限不足",
}
