# 工地人员识别与安全监控系统（桌面端）

**中国大学生计算机设计大赛**作品：单机桌面工作站，集成人员管理、摄像头监控、人脸与安全帽/吸烟检测、考勤与违规等业务。

## 运行方式（仅需桌面端）

```powershell
cd 本项目目录
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_security_system.py
```

说明：

- 请始终在**项目根目录**下运行（脚本会自动 `chdir` 到项目根，保证 `security_system.db`、人脸样本、模型路径一致）。
- 监控：主界面顶栏右侧 **「启动监控」**，在 **「门禁与监控」** 标签中查看画面（逻辑与此前版本一致）。

## 代码结构（规范分层）

| 模块 | 职责 |
|------|------|
| `run_security_system.py` | 进程入口、依赖检查 |
| `imsafe/desktop_app.py` | 业务与监控主程序（Tk + OpenCV） |
| `imsafe/desktop_config.py` | 窗口标题、区域/权限等**常量配置** |
| `imsafe/ui/theme.py` | **外观主题**（ttk 样式，与业务分离） |
| `imsafe/database.py` / `paths.py` | 数据库路径与建表 |
| `imsafe/ai_framework.py` | 可插拔视觉算法服务（默认模拟推理） |

详细设计说明见：`docs/软件设计方案_中国大学生计算机设计大赛.md`

## 测试

```powershell
python test_simple.py
```
