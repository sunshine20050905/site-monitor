
"""
工地人员识别与安全监控系统 — 桌面端启动脚本
中国大学生计算机设计大赛（4C）作品方向：工地人脸识别与安全监控

功能概要：
人员、考勤、违规、工资等业务（SQLite）
OpenCV 摄像头监控、LBPH 人脸、安全帽/吸烟检测（可扩展 ONNX）
可插拔「工地视觉算法服务」（默认 CPU 模拟，见 imsafe/ai_framework.py）
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import traceback


def check_dependencies():
    """检查依赖项"""
    required_modules = [
        "cv2",
        "numpy",
        "sqlite3",
        "tkinter",
        "matplotlib",
        "PIL",
        "threading",
        "time",
        "winsound",
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print("❌ 缺少以下依赖模块:")
        for module in missing_modules:
            print(f"   - {module}")
        here = os.path.dirname(os.path.abspath(__file__))
        print("\n请在项目目录下使用官方 Python 的虚拟环境安装依赖，例如：")
        print("  py -3.12 -m venv .venv")
        print(r"  .\.venv\Scripts\Activate.ps1")
        print("  pip install -r requirements.txt")
        print("\n当前脚本目录:", here)
        return False

    return True


def main():
    """主函数"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("🚀 启动工地人员识别与安全监控系统（中国大学生计算机设计大赛）")
    print("=" * 60)

    print("📦 检查依赖项...")
    if not check_dependencies():
        return False

    print("✅ 所有依赖项检查通过")

    try:
        print("📋 导入系统模块...")
        from imsafe import SecurityManagementSystem

        print("🖥️ 创建主窗口...")
        root = tk.Tk()

        try:
            pass
        except Exception:
            pass

        print("⚙️ 初始化系统...")
        app = SecurityManagementSystem(root)

        print("✅ 系统初始化完成")
        print("🎯 启动图形界面...")

        messagebox.showinfo(
            "系统启动",
            "工地人员识别与安全监控系统\n（中国大学生计算机设计大赛）\n\n"
            "✅ 系统启动成功\n"
            "🤖 可插拔视觉算法服务已就绪（默认模拟推理）\n"
            "🔍 人脸识别、安全帽与吸烟检测\n"
            "📊 考勤、违规与统计报表\n"
            "🎓 「算法与训练」页可查看演示训练任务\n",
        )

        root.mainloop()

        return True

    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保所有文件都在同一目录下")
        return False

    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()

        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "系统启动失败",
                f"系统启动时发生错误：\n\n{e}\n\n"
                "请检查系统配置或依赖是否已安装",
            )
        except Exception:
            pass

        return False


def show_help():
    """显示帮助信息"""
    help_text = """
工地人员识别与安全监控系统 — 中国大学生计算机设计大赛

🎯 主要功能：
- 人员信息管理
- 人脸识别与区域权限
- 安全帽、吸烟等行为检测
- 考勤、违规、工资与统计
- 算法与训练（演示调度器，可对接真实训练流程）

📋 使用说明：
1. 「人员管理」添加人员并采集人脸样本，训练 LBPH 模型
2. 「门禁管理」启动摄像头监控
3. 查看考勤、违规与统计
4. 「算法与训练」查看推理服务状态与演示训练任务

🔧 系统要求：
- Python 3.10+（推荐 3.12）
- OpenCV contrib、NumPy、Tkinter、Matplotlib、Pillow
    """

    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("系统帮助", help_text)
    except Exception:
        print(help_text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_help()
    else:
        success = main()
        sys.exit(0 if success else 1)
