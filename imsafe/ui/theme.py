"""
桌面端 ttk 主题：统一配色、字体与控件样式。
仅影响外观，不改变监控线程与 OpenCV 逻辑。
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def apply_desktop_theme(root: tk.Tk, style: ttk.Style) -> dict:
    """
    应用主题并返回颜色字典，供 tk.Frame / matplotlib 等需要处使用。
    """
    style.theme_use("clam")

    # 高端大气配色方案 - 深蓝与灰色调
    bg = "#f8fafc"  # 浅灰白背景
    card = "#ffffff"  # 纯白卡片
    text = "#1e293b"  # 深蓝文字
    muted = "#64748b"  # 灰色次要文字
    header = "#1e40af"  # 深蓝头部
    header_accent = "#3b82f6"  # 亮蓝强调
    accent = "#2563eb"  # 蓝色主要按钮
    accent_hover = "#1d4ed8"  # 深蓝悬停
    border = "#e2e8f0"  # 浅灰边框
    tab_idle = "#f1f5f9"  # 标签页空闲
    select_bg = "#dbeafe"  # 选中背景
    success = "#059669"  # 绿色成功
    success_hover = "#047857"  # 深绿悬停
    danger = "#dc2626"  # 红色危险
    danger_hover = "#b91c1c"  # 深红悬停
    warning = "#d97706"  # 橙色警告
    info = "#0891b2"  # 青色信息

    colors = {
        "bg": bg,
        "card": card,
        "text": text,
        "muted": muted,
        "header": header,
        "header_accent": header_accent,
        "accent": accent,
        "accent_hover": accent_hover,
        "border": border,
        "tab_idle": tab_idle,
        "select_bg": select_bg,
        "success": success,
        "success_hover": success_hover,
        "danger": danger,
        "danger_hover": danger_hover,
        "warning": warning,
        "info": info,
    }

    root.configure(bg=bg)

    # 全局样式
    style.configure(".", background=bg, foreground=text, font=("Microsoft YaHei UI", 10))
    style.configure("TLabel", background=bg, foreground=text, font=("Microsoft YaHei UI", 10))
    style.configure("TFrame", background=bg)
    style.configure("Card.TFrame", background=card, relief="flat", borderwidth=1)

    # 头部样式 - 高端深蓝
    style.configure(
        "Header.TFrame",
        background=header,
    )
    style.configure(
        "HeaderTitle.TLabel",
        background=header,
        foreground="#ffffff",
        font=("Microsoft YaHei UI", 16, "bold"),
    )
    style.configure(
        "HeaderSub.TLabel",
        background=header,
        foreground="#e0e7ff",
        font=("Microsoft YaHei UI", 10),
    )
    style.configure(
        "HeaderStatus.TLabel",
        background=header,
        foreground="#fecaca",
        font=("Microsoft YaHei UI", 11, "bold"),
    )
    style.configure(
        "HeaderStatusOn.TLabel",
        background=header,
        foreground="#bbf7d0",
        font=("Microsoft YaHei UI", 11, "bold"),
    )

    # 标签框架 - 现代化设计
    style.configure(
        "TLabelframe",
        background=card,
        foreground=header,
        bordercolor=border,
        relief="solid",
        borderwidth=2,
        lightcolor=border,
        darkcolor=border,
    )
    style.configure(
        "TLabelframe.Label",
        background=card,
        foreground=header,
        font=("Microsoft YaHei UI", 11, "bold"),
    )

    # 按钮样式 - 现代化圆角效果
    style.configure(
        "TButton",
        background=accent,
        foreground="#ffffff",
        font=("Microsoft YaHei UI", 10, "bold"),
        borderwidth=0,
        relief="flat",
        padding=(10, 5),
    )
    style.map(
        "TButton",
        background=[("active", accent_hover), ("pressed", accent_hover)],
        relief=[("pressed", "sunken")],
    )

    # 成功按钮
    style.configure(
        "Success.TButton",
        background=success,
        foreground="#ffffff",
        font=("Microsoft YaHei UI", 10, "bold"),
    )
    style.map(
        "Success.TButton",
        background=[("active", success_hover), ("pressed", success_hover)],
    )

    # 危险按钮
    style.configure(
        "Danger.TButton",
        background=danger,
        foreground="#ffffff",
        font=("Microsoft YaHei UI", 10, "bold"),
    )
    style.map(
        "Danger.TButton",
        background=[("active", danger_hover), ("pressed", danger_hover)],
    )

    # 标签页样式
    style.configure(
        "TNotebook",
        background=bg,
        borderwidth=0,
    )
    style.configure(
        "TNotebook.Tab",
        background=tab_idle,
        foreground=text,
        font=("Microsoft YaHei UI", 10),
        padding=(15, 8),
        borderwidth=1,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", card)],
        foreground=[("selected", header)],
        font=[("selected", ("Microsoft YaHei UI", 10, "bold"))],
    )

    # 输入框样式
    style.configure(
        "TEntry",
        fieldbackground=card,
        bordercolor=border,
        lightcolor=border,
        darkcolor=border,
        insertcolor=text,
        font=("Microsoft YaHei UI", 10),
    )

    # 组合框样式
    style.configure(
        "TCombobox",
        fieldbackground=card,
        background=card,
        bordercolor=border,
        arrowcolor=text,
        font=("Microsoft YaHei UI", 10),
    )

    # 树视图样式
    style.configure(
        "Treeview",
        background=card,
        foreground=text,
        fieldbackground=card,
        bordercolor=border,
        font=("Microsoft YaHei UI", 10),
    )
    style.configure(
        "Treeview.Heading",
        background=header,
        foreground="#ffffff",
        font=("Microsoft YaHei UI", 10, "bold"),
    )
    style.map(
        "Treeview.Heading",
        background=[("active", header_accent)],
    )

    # 滚动条样式
    style.configure(
        "TScrollbar",
        background=border,
        troughcolor=bg,
        borderwidth=0,
        arrowcolor=text,
    )
    style.map(
        "TScrollbar",
        background=[("active", accent)],
    )

    # 单选按钮样式
    style.configure(
        "TRadiobutton",
        background=bg,
        foreground=text,
        font=("Microsoft YaHei UI", 10),
    )

    # 底部状态栏
    style.configure(
        "Footer.TFrame",
        background=muted,
    )
    style.configure(
        "Footer.TLabel",
        background=muted,
        foreground="#ffffff",
        font=("Microsoft YaHei UI", 9),
    )

    return colors

    style.configure(
        "Monitor.TButton",
        background=success,
        foreground="#ffffff",
        font=("Microsoft YaHei UI", 11, "bold"),
        padding=(18, 9),
    )
    style.map(
        "Monitor.TButton",
        background=[("active", success_hover), ("pressed", success_hover)],
    )

    style.configure(
        "StopMonitor.TButton",
        background=danger,
        foreground="#ffffff",
        font=("Microsoft YaHei UI", 11, "bold"),
        padding=(18, 9),
    )
    style.map(
        "StopMonitor.TButton",
        background=[("active", danger_hover), ("pressed", danger_hover)],
    )

    style.configure("Footer.TFrame", background="#e2e8f0", relief="flat")
    style.configure(
        "Footer.TLabel",
        background="#e2e8f0",
        foreground=muted,
        font=("Microsoft YaHei UI", 9),
    )

    return colors
