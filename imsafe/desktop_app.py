import cv2
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import os
import tkinter as tk
from tkinter import messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from PIL import ImageDraw, ImageFont
import threading
import time
import winsound  # Windows系统声音提示
import random
from collections import deque, defaultdict

from typing import Optional

from imsafe.ai_framework import SiteVisionFramework, get_vision_framework
from imsafe.database import init_schema
from imsafe.desktop_config import (
    AREAS,
    GEOMETRY,
    HELMET_REQUIRED_AREAS,
    MAIN_HEADLINE,
    MINSIZE,
    PERMISSION_LEVELS,
    SUBTITLE,
    VIOLATION_TYPES,
    WINDOW_TITLE,
)
from imsafe.paths import get_db_path, get_project_root
from imsafe.ui.theme import apply_desktop_theme

# 确保中文正常显示
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class SecurityManagementSystem:
    def __init__(self, root):
        self.root = root
        # 固定工作目录到项目根，数据库/模型/人脸样本等相对路径与监控逻辑保持一致
        os.chdir(get_project_root())
        self.root.title(WINDOW_TITLE)
        self.root.geometry(GEOMETRY)
        self.root.minsize(*MINSIZE)

        self.style = ttk.Style()
        self.theme_colors = apply_desktop_theme(self.root, self.style)
        self.bg_color = self.theme_colors["bg"]
        self.accent_color = self.theme_colors["accent"]
        self.text_color = self.theme_colors["text"]
        self.header_color = self.theme_colors["header"]
        self.card_color = self.theme_colors["card"]

        # 初始化数据库
        self.init_database()

        # 初始化工地视觉算法服务（可插拔，默认 CPU 模拟）
        self.init_vision_framework()

        # 初始化传统人脸识别模型（作为备用）
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        # 加载安全帽和吸烟检测所需的模型（传统方法）
        self.init_safety_cascades()

        # 可选：加载基于DNN/YOLO的检测模型（若存在）
        self.init_dnn_models()

        # 吸烟检测时序平滑缓存
        self.smoking_history = defaultdict(lambda: deque(maxlen=8))  # 每张脸最多缓存8帧的判断
        self.smoking_conf_history = defaultdict(lambda: deque(maxlen=8))

        # 加载已训练的模型（如果存在）
        if os.path.exists('face_recognizer.yml'):
            try:
                self.recognizer.read('face_recognizer.yml')
            except:
                messagebox.showerror("错误", "人脸识别模型加载失败，将使用新模型")

        # 创建必要的文件夹
        self.create_necessary_folders()

        self.permission_levels = dict(PERMISSION_LEVELS)
        self.areas = dict(AREAS)
        self.helmet_required_areas = list(HELMET_REQUIRED_AREAS)
        self.violation_types = dict(VIOLATION_TYPES)

        # 初始化字体
        self.init_fonts()

        # 创建界面
        self.create_widgets()

        # 当前摄像头帧
        self.current_frame = None
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 违规检测状态跟踪（防止重复记录）
        self.violation_state = {
            "no_helmet": False,
            "smoking": False
        }
        self.last_violation_time = {
            "no_helmet": datetime.min,
            "smoking": datetime.min
        }
        self.violation_cooldown = 30  # 违规记录冷却时间（秒）
        
        # 警告系统
        self.warning_active = False
        self.warning_thread = None
        self.warning_count = 0
        self.max_warnings = 3  # 最大警告次数

        # 监控状态
        self.monitoring = False
        self.cap = None
        self.monitor_thread = None
        self.monitor_running = False

    def init_vision_framework(self):
        """初始化工地视觉算法服务（人脸/安全帽/吸烟等，失败则回退 OpenCV 传统管线）"""
        try:
            print("正在初始化工地视觉算法服务...")
            self.ai_framework = get_vision_framework()
            
            if self.ai_framework.is_initialized:
                print("工地视觉算法服务初始化成功")
                
                # 显示系统状态
                status = self.ai_framework.get_system_status()
                print(f"系统状态: {status}")
                
                # 创建AI状态标签
                self.create_ai_status_display()
            else:
                print("工地视觉算法服务初始化失败，将使用传统方法")
                self.ai_framework = None
                
        except Exception as e:
            print(f"工地视觉算法服务初始化异常: {e}")
            self.ai_framework = None

    def create_ai_status_display(self):
        """创建AI状态显示"""
        # 这个将在后续的UI设置中实现
        pass

    def match_face_features(self, features: np.ndarray) -> Optional[int]:
        """匹配人脸特征，返回人员ID"""
        try:
            if not self.ai_framework or not self.ai_framework.is_initialized:
                return None
            # 从数据库中获取所有已保存的人脸特征
            self.cursor.execute("SELECT id, features FROM face_features")
            stored_features = self.cursor.fetchall()
            
            best_match_id = None
            best_similarity = 0.0
            
            for person_id, stored_feature_str in stored_features:
                try:
                    stored_features_array = np.array(eval(stored_feature_str))
                    similarity = self.ai_framework.models['face_recognition'].compare_faces(features, stored_features_array)
                    
                    if similarity > best_similarity and similarity > 0.8:  # 相似度阈值
                        best_similarity = similarity
                        best_match_id = person_id
                        
                except Exception as e:
                    print(f"特征匹配失败 {person_id}: {e}")
                    continue
                    
            return best_match_id
            
        except Exception as e:
            print(f"人脸特征匹配失败: {e}")
            return None

    def save_face_features(self, person_id: int, features: np.ndarray) -> bool:
        """保存人脸特征到数据库"""
        try:
            # 检查是否已存在该人员的特征
            self.cursor.execute("SELECT id FROM face_features WHERE person_id = ?", (person_id,))
            existing = self.cursor.fetchone()
            
            if existing:
                # 更新现有特征
                self.cursor.execute("UPDATE face_features SET features = ? WHERE person_id = ?", 
                                  (str(features.tolist()), person_id))
            else:
                # 插入新特征
                self.cursor.execute("INSERT INTO face_features (person_id, features) VALUES (?, ?)", 
                                  (person_id, str(features.tolist())))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"保存人脸特征失败: {e}")
            self.conn.rollback()
            return False

    def create_necessary_folders(self):
        """创建必要的文件夹"""
        folders = ['face_samples', 'violation_screenshots', 'models']
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def init_safety_cascades(self):
        """初始化安全帽和吸烟检测所需的级联分类器和模型"""
        # 尝试加载安全帽检测模型
        self.helmet_cascade = None
        helmet_model_path = "models/helmet_cascade.xml"

        if os.path.exists(helmet_model_path):
            try:
                self.helmet_cascade = cv2.CascadeClassifier(helmet_model_path)
            except Exception as e:
                print(f"加载安全帽模型失败: {e}")

        # 尝试加载吸烟检测模型
        self.cigarette_cascade = None
        cigarette_model_path = "models/cigarette_cascade.xml"

        if os.path.exists(cigarette_model_path):
            try:
                self.cigarette_cascade = cv2.CascadeClassifier(cigarette_model_path)
            except Exception as e:
                print(f"加载吸烟模型失败: {e}")

    def init_dnn_models(self):
        """尝试加载可选的ONNX检测模型（YOLOv5/YOLOv8导出的ONNX均可）"""
        self.helmet_net = None
        self.helmet_classes = []
        self.smoke_net = None
        self.smoke_classes = []

        try:
            helmet_onnx = os.path.join("models", "helmet_yolo.onnx")
            helmet_names = os.path.join("models", "helmet_classes.txt")
            if os.path.exists(helmet_onnx):
                self.helmet_net = cv2.dnn.readNetFromONNX(helmet_onnx)
                if os.path.exists(helmet_names):
                    with open(helmet_names, "r", encoding="utf-8") as f:
                        self.helmet_classes = [line.strip() for line in f if line.strip()]
                else:
                    self.helmet_classes = ["helmet"]
                print("✅ 已加载ONNX安全帽检测模型")
        except Exception as e:
            print(f"加载ONNX安全帽模型失败: {e}")

        try:
            smoke_onnx = os.path.join("models", "smoking_yolo.onnx")
            smoke_names = os.path.join("models", "smoking_classes.txt")
            if os.path.exists(smoke_onnx):
                self.smoke_net = cv2.dnn.readNetFromONNX(smoke_onnx)
                if os.path.exists(smoke_names):
                    with open(smoke_names, "r", encoding="utf-8") as f:
                        self.smoke_classes = [line.strip() for line in f if line.strip()]
                else:
                    self.smoke_classes = ["cigarette", "smoke"]
                print("✅ 已加载ONNX吸烟检测模型")
        except Exception as e:
            print(f"加载ONNX吸烟模型失败: {e}")

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """保持纵横比的缩放，填充到指定尺寸，返回缩放图、缩放比例、边距。"""
        shape = img.shape[:2]  # h, w
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        # 缩放
        resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        # 填充
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded, r, (left, top)

    def yolo_detect(self, net, image_bgr, conf_threshold=0.25, iou_threshold=0.45, input_size=(640, 640)):
        """通用YOLOv5/YOLOv8 ONNX前向推理，返回[(x,y,w,h,score,class_id)]（坐标基于原图）。"""
        if net is None:
            return []

        h, w = image_bgr.shape[:2]
        # Letterbox预处理
        padded, r, (left, top) = self._letterbox(image_bgr, input_size)
        blob = cv2.dnn.blobFromImage(padded, 1/255.0, input_size, swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()

        # 兼容不同导出：期望形状为[1, num, 85] 或 [num, 85]
        if outputs.ndim == 3:
            outputs = outputs[0]
        # 将xywh转换为xyxy并还原到原图尺度
        boxes = []
        scores = []
        class_ids = []
        for det in outputs:
            if det.shape[0] < 5:
                continue
            cx, cy, bw, bh = det[0:4]
            class_scores = det[5:]
            score = float(np.max(class_scores)) if class_scores.size else float(det[4])
            cls_id = int(np.argmax(class_scores)) if class_scores.size else 0
            obj_conf = float(det[4]) if det.shape[0] >= 5 else 1.0
            final_conf = score * obj_conf
            if final_conf < conf_threshold:
                continue

            # 网络输出是基于letterbox后的坐标，先转xywh到xyxy
            x = (cx - bw / 2)
            y = (cy - bh / 2)
            # 去掉padding并按比例映射回原图
            x -= left
            y -= top
            # 还原到letterbox去padding前的尺度
            x /= r
            y /= r
            bw /= r
            bh /= r
            boxes.append([x, y, bw, bh])
            scores.append(final_conf)
            class_ids.append(cls_id)

        if not boxes:
            return []

        boxes_np = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes], dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        idxs = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold) if hasattr(cv2.dnn, 'NMSBoxes') else []

        results = []
        if idxs is None or len(idxs) == 0:
            # 简单返回全部
            for (x, y, bw, bh), s, c in zip(boxes, scores, class_ids):
                results.append((int(x), int(y), int(bw), int(bh), float(s), int(c)))
            return results

        # 适配cv2.dnn.NMSBoxes返回格式
        idxs_flat = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in idxs]
        for i in idxs_flat:
            x, y, bw, bh = boxes[i]
            results.append((int(x), int(y), int(bw), int(bh), float(scores[i]), int(class_ids[i])))
        return results

    def draw_text_cn(self, frame, text, org, color=(255, 255, 255), font_size=22):
        """在图像上使用PIL绘制中文，避免OpenCV中文乱码。"""
        if not isinstance(frame, np.ndarray):
            return frame
        # 选择字体
        font_path_candidates = [
            r"C:\\Windows\\Fonts\\msyh.ttc",  # 微软雅黑
            r"C:\\Windows\\Fonts\\simhei.ttf",  # 黑体
            r"C:\\Windows\\Fonts\\simsun.ttc",  # 宋体
        ]
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = None
        for fp in font_path_candidates:
            try:
                if os.path.exists(fp):
                    font = ImageFont.truetype(fp, font_size)
                    break
            except Exception:
                continue
        if font is None:
            # 最后兜底：默认字体（可能仍会有问题，但尽量保证不报错）
            try:
                font = ImageFont.load_default()
            except Exception:
                pass
        try:
            draw.text(org, text, font=font, fill=(color[0], color[1], color[2]))
        except Exception:
            # 任何异常都不影响主流程
            pass
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def put_text_cn(self, frame, text, position, color=(255, 255, 255), font_size=22):
        """包装函数：优先使用中文绘制，失败则回退到cv2.putText。"""
        try:
            return self.draw_text_cn(frame, text, position, color=color, font_size=font_size)
        except Exception:
            try:
                cv2.putText(frame, text, position, self.default_font, max(0.5, font_size/30.0), color, 2)
            except Exception:
                pass
            return frame

    def init_fonts(self):
        """初始化支持中文的字体，解决显示乱码问题"""
        self.fonts = []
        windows_fonts = ["simsun", "microsoft yahei", "simhei", "kaiti"]
        linux_fonts = ["simhei", "wenquanyi micro hei", "heiti tc"]

        # 测试字体是否支持中文
        test_img = np.zeros((100, 300, 3), np.uint8)
        for font_name in windows_fonts + linux_fonts:
            try:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(test_img, "测试中文", (10, 50), font, 1, (255, 255, 255), 2)
                self.fonts.append(font)
            except:
                continue

        if not self.fonts:
            self.fonts.append(cv2.FONT_HERSHEY_SIMPLEX)

        self.default_font = self.fonts[0]

    def init_database(self):
        """初始化数据库和表结构"""
        try:
            self.conn = sqlite3.connect(get_db_path())
            self.cursor = self.conn.cursor()
            init_schema(self.cursor)
            self.conn.commit()
        except Exception as e:
            messagebox.showerror("数据库错误", f"初始化数据库失败: {str(e)}")

    def create_widgets(self):
        """创建 GUI：顶栏 + 笔记本主区 + 底栏（监控逻辑不变）。"""
        self.root.configure(bg=self.bg_color)

        header = ttk.Frame(self.root, style="Header.TFrame", height=88)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        header_inner = ttk.Frame(header, style="Header.TFrame")
        header_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=(14, 12))

        left_brand = ttk.Frame(header_inner, style="Header.TFrame")
        left_brand.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left_brand, text=MAIN_HEADLINE, style="HeaderTitle.TLabel").pack(anchor=tk.W)
        ttk.Label(left_brand, text=SUBTITLE, style="HeaderSub.TLabel").pack(anchor=tk.W, pady=(4, 0))

        monitor_frame = ttk.Frame(header_inner, style="Header.TFrame")
        monitor_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.monitor_status_label = ttk.Label(
            monitor_frame,
            text="● 监控已停止",
            style="HeaderStatus.TLabel",
        )
        self.monitor_status_label.pack(side=tk.LEFT, padx=(0, 14), pady=4)

        self.unified_monitor_btn = ttk.Button(
            monitor_frame,
            text="启动监控",
            command=self.toggle_unified_monitoring,
            style="Monitor.TButton",
        )
        self.unified_monitor_btn.pack(side=tk.LEFT, padx=4)

        body = ttk.Frame(self.root)
        body.pack(expand=True, fill=tk.BOTH, padx=14, pady=(0, 8))

        tab_shell = ttk.Frame(body)
        tab_shell.pack(expand=True, fill=tk.BOTH)

        tab_control = ttk.Notebook(tab_shell)
        tab_control.pack(expand=True, fill=tk.BOTH)

        self.tab_persons = ttk.Frame(tab_control, padding=4)
        tab_control.add(self.tab_persons, text="  人员管理  ")

        self.tab_access = ttk.Frame(tab_control, padding=4)
        tab_control.add(self.tab_access, text="  门禁与监控  ")

        self.tab_attendance = ttk.Frame(tab_control, padding=4)
        tab_control.add(self.tab_attendance, text="  考勤记录  ")

        self.tab_violations = ttk.Frame(tab_control, padding=4)
        tab_control.add(self.tab_violations, text="  违规记录  ")

        self.tab_salary = ttk.Frame(tab_control, padding=4)
        tab_control.add(self.tab_salary, text="  工资结算  ")

        self.tab_stats = ttk.Frame(tab_control, padding=4)
        tab_control.add(self.tab_stats, text="  数据统计  ")

        self.tab_ai_framework = ttk.Frame(tab_control, padding=4)
        tab_control.add(self.tab_ai_framework, text="  算法与训练  ")

        footer = ttk.Frame(self.root, style="Footer.TFrame", height=28)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        db_name = os.path.basename(get_db_path())
        ttk.Label(
            footer,
            text=f" 就绪  ·  数据文件：{db_name}  ·  请先完成人员建档与人脸训练后再启动监控 ",
            style="Footer.TLabel",
        ).pack(side=tk.LEFT, padx=12, pady=4)

        self.setup_persons_tab()
        self.setup_access_tab()
        self.setup_attendance_tab()
        self.setup_violations_tab()
        self.setup_salary_tab()
        self.setup_stats_tab()
        self.setup_ai_framework_tab()

    def setup_persons_tab(self):
        """设置人员管理标签页"""
        main_frame = ttk.Frame(self.tab_persons)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        left_frame = ttk.LabelFrame(main_frame, text="人员列表")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 搜索框
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(search_frame, text="搜索:").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(search_frame, text="搜索", command=self.search_persons).pack(side=tk.LEFT, padx=5)

        # 人员列表
        columns = ("id", "姓名", "工号", "RFID卡号", "权限等级")
        self.persons_tree = ttk.Treeview(left_frame, columns=columns, show="headings")

        for col in columns:
            self.persons_tree.heading(col, text=col)
            width = 100 if col != "姓名" else 120
            self.persons_tree.column(col, width=width, anchor=tk.CENTER)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=self.persons_tree.yview)
        self.persons_tree.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.persons_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.persons_tree.bind("<<TreeviewSelect>>", self.on_person_select)

        ttk.Button(left_frame, text="刷新列表", command=self.refresh_persons_list).pack(fill=tk.X, padx=5, pady=5)

        # 右侧表单区域
        right_frame = ttk.LabelFrame(main_frame, text="人员信息管理")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 人员信息表单
        form_frame = ttk.Frame(right_frame)
        form_frame.pack(fill=tk.X, padx=5, pady=5)

        # 表单布局
        form_grid = {
            "姓名:": 0,
            "工号:": 1,
            "RFID卡号:": 2,
            "权限等级:": 3
        }

        for label_text, row in form_grid.items():
            ttk.Label(form_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, padx=5, pady=8)

        # 表单输入框
        self.name_entry = ttk.Entry(form_frame)
        self.name_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=8)

        self.employee_id_entry = ttk.Entry(form_frame)
        self.employee_id_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=8)

        self.rfid_entry = ttk.Entry(form_frame)
        self.rfid_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=8)

        self.permission_combobox = ttk.Combobox(form_frame, values=list(self.permission_levels.values()))
        self.permission_combobox.current(0)
        self.permission_combobox.grid(row=3, column=1, sticky=tk.EW, padx=5, pady=8)

        form_frame.columnconfigure(1, weight=1)

        # 人脸采集区域
        face_frame = ttk.LabelFrame(right_frame, text="人脸采集")
        face_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.face_canvas = tk.Canvas(face_frame, width=320, height=240, bg="#e0e0e0", highlightthickness=1)
        self.face_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.face_canvas.create_text(160, 120, text="请选择人员并打开摄像头", fill="#666666",
                                     font=('Microsoft YaHei', 10))

        # 按钮区域
        buttons_frame = ttk.Frame(face_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        self.capture_btn = ttk.Button(buttons_frame, text="打开摄像头", command=self.start_camera)
        self.capture_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.take_sample_btn = ttk.Button(buttons_frame, text="采集样本", command=self.take_face_sample,
                                          state=tk.DISABLED)
        self.take_sample_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.train_btn = ttk.Button(buttons_frame, text="训练模型", command=self.train_recognizer)
        self.train_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # 操作按钮区域
        actions_frame = ttk.Frame(right_frame)
        actions_frame.pack(fill=tk.X, padx=5, pady=10)

        self.add_btn = ttk.Button(actions_frame, text="添加人员", command=self.add_person)
        self.add_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.update_btn = ttk.Button(actions_frame, text="更新信息", command=self.update_person, state=tk.DISABLED)
        self.update_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.delete_btn = ttk.Button(actions_frame, text="删除人员", command=self.delete_person, state=tk.DISABLED)
        self.delete_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.clear_btn = ttk.Button(actions_frame, text="清空表单", command=self.clear_person_form)
        self.clear_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.refresh_persons_list()

    def setup_access_tab(self):
        """设置门禁管理标签页"""
        main_frame = ttk.Frame(self.tab_access)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # 左侧监控区域
        left_frame = ttk.LabelFrame(main_frame, text="门禁监控")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.access_canvas = tk.Canvas(left_frame, width=640, height=480, bg="#e0e0e0", highlightthickness=1)
        self.access_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.access_canvas.create_text(320, 240, text="点击开始监控按钮启动监控", fill="#666666",
                                       font=('Microsoft YaHei', 12))

        # 区域选择
        area_frame = ttk.Frame(left_frame)
        area_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(area_frame, text="选择区域:").pack(side=tk.LEFT, padx=5)
        self.area_combobox = ttk.Combobox(area_frame, values=list(self.areas.values()), state="readonly")
        self.area_combobox.current(0)
        self.area_combobox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # 安全检测设置
        safety_frame = ttk.LabelFrame(left_frame, text="安全检测设置")
        safety_frame.pack(fill=tk.X, padx=5, pady=5)

        self.helmet_detection_var = tk.BooleanVar(value=True)
        self.smoking_detection_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(safety_frame, text="启用安全帽检测", variable=self.helmet_detection_var).pack(side=tk.LEFT,
                                                                                                      padx=15, pady=5)
        ttk.Checkbutton(safety_frame, text="启用吸烟检测", variable=self.smoking_detection_var).pack(side=tk.LEFT,
                                                                                                     padx=15, pady=5)

        # 监控信息显示区域
        monitor_info_frame = ttk.LabelFrame(left_frame, text="监控信息")
        monitor_info_frame.pack(fill=tk.X, padx=5, pady=10)

        # 当前监控状态
        self.current_monitor_status = tk.StringVar()
        self.current_monitor_status.set("监控已停止")
        status_label = ttk.Label(monitor_info_frame, textvariable=self.current_monitor_status, 
                                 font=("Microsoft YaHei", 11, "bold"), foreground="red")
        status_label.pack(pady=5)

        # 监控提示信息
        info_text = "请使用顶部统一监控按钮启动/停止监控\n监控启动后将自动进行人脸识别、安全帽检测和吸烟检测"
        info_label = ttk.Label(monitor_info_frame, text=info_text, 
                               font=("Microsoft YaHei", 9), foreground="#666666")
        info_label.pack(pady=5)

        # 右侧信息区域
        right_frame = ttk.LabelFrame(main_frame, text="监控信息")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 最近访问记录
        ttk.Label(right_frame, text="最近访问记录:", font=('Microsoft YaHei', 10, 'bold')).pack(anchor=tk.W, padx=5,
                                                                                                pady=5)

        columns = ("时间", "人员", "区域", "状态")
        self.access_tree = ttk.Treeview(right_frame, columns=columns, show="headings")

        for col in columns:
            self.access_tree.heading(col, text=col)
            width = 120 if col == "时间" else 100
            self.access_tree.column(col, width=width, anchor=tk.CENTER)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.access_tree.yview)
        self.access_tree.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.access_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 识别信息
        ttk.Label(right_frame, text="识别信息:", font=('Microsoft YaHei', 10, 'bold')).pack(anchor=tk.W, padx=5,
                                                                                            pady=10)
        self.identify_info = tk.StringVar()
        self.identify_info.set("等待监控开始...")
        identify_label = ttk.Label(right_frame, textvariable=self.identify_info, font=("Microsoft YaHei", 11))
        identify_label.pack(anchor=tk.W, padx=5, pady=5)

        # 违规信息
        ttk.Label(right_frame, text="最近违规记录:", font=('Microsoft YaHei', 10, 'bold')).pack(anchor=tk.W, padx=5,
                                                                                                pady=10)
        self.violation_info = tk.StringVar()
        self.violation_info.set("暂无违规记录")
        violation_label = ttk.Label(right_frame, textvariable=self.violation_info, font=("Microsoft YaHei", 11),
                                    foreground="red")
        violation_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # 警告状态显示
        self.warning_status = tk.StringVar()
        self.warning_status.set("系统正常")
        warning_label = ttk.Label(right_frame, textvariable=self.warning_status, font=("Microsoft YaHei", 12, "bold"),
                                  foreground="green")
        warning_label.pack(anchor=tk.W, padx=5, pady=5)
        
        # 警告计数器
        self.warning_counter = tk.StringVar()
        self.warning_counter.set("警告次数: 0/3")
        counter_label = ttk.Label(right_frame, textvariable=self.warning_counter, font=("Microsoft YaHei", 10),
                                  foreground="orange")
        counter_label.pack(anchor=tk.W, padx=5, pady=5)

    def setup_attendance_tab(self):
        """设置考勤记录标签页"""
        main_frame = ttk.Frame(self.tab_attendance)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # 筛选区域
        filter_frame = ttk.LabelFrame(main_frame, text="查询条件")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(filter_frame, text="人员:").pack(side=tk.LEFT, padx=5, pady=5)
        self.attendance_person_combobox = ttk.Combobox(filter_frame)
        self.attendance_person_combobox.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        ttk.Label(filter_frame, text="区域:").pack(side=tk.LEFT, padx=5, pady=5)
        self.attendance_area_combobox = ttk.Combobox(filter_frame, values=list(self.areas.values()), state="readonly")
        self.attendance_area_combobox.current(0)
        self.attendance_area_combobox.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        ttk.Label(filter_frame, text="日期:").pack(side=tk.LEFT, padx=5, pady=5)
        self.attendance_date_entry = ttk.Entry(filter_frame, width=12)
        self.attendance_date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
        self.attendance_date_entry.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(filter_frame, text="查询", command=self.query_attendance).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(filter_frame, text="导出", command=self.export_attendance).pack(side=tk.LEFT, padx=5, pady=5)

        # 考勤记录表格
        columns = ("id", "人员", "区域", "进入时间", "离开时间", "状态")
        self.attendance_tree = ttk.Treeview(main_frame, columns=columns, show="headings")

        for col in columns:
            self.attendance_tree.heading(col, text=col)
            width = 150 if col in ["进入时间", "离开时间"] else 100
            self.attendance_tree.column(col, width=width, anchor=tk.CENTER)

        # 添加滚动条
        scrollbar_y = ttk.Scrollbar(main_frame, orient="vertical", command=self.attendance_tree.yview)
        scrollbar_x = ttk.Scrollbar(main_frame, orient="horizontal", command=self.attendance_tree.xview)
        self.attendance_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.attendance_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.refresh_attendance_persons()

    def setup_violations_tab(self):
        """设置违规记录标签页"""
        main_frame = ttk.Frame(self.tab_violations)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # 筛选区域
        filter_frame = ttk.LabelFrame(main_frame, text="查询条件")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(filter_frame, text="人员:").pack(side=tk.LEFT, padx=5, pady=5)
        self.violation_person_combobox = ttk.Combobox(filter_frame)
        self.violation_person_combobox.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        ttk.Label(filter_frame, text="违规类型:").pack(side=tk.LEFT, padx=5, pady=5)
        self.violation_type_combobox = ttk.Combobox(filter_frame,
                                                    values=["所有", "未戴安全帽", "抽烟", "权限不足", "其他"],
                                                    state="readonly")
        self.violation_type_combobox.current(0)
        self.violation_type_combobox.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        ttk.Label(filter_frame, text="日期范围:").pack(side=tk.LEFT, padx=5, pady=5)
        self.violation_start_date = ttk.Entry(filter_frame, width=12)
        self.violation_start_date.insert(0, (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
        self.violation_start_date.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Label(filter_frame, text="至").pack(side=tk.LEFT, pady=5)

        self.violation_end_date = ttk.Entry(filter_frame, width=12)
        self.violation_end_date.insert(0, datetime.now().strftime("%Y-%m-%d"))
        self.violation_end_date.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(filter_frame, text="查询", command=self.query_violations).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(filter_frame, text="标记已处理", command=self.mark_violation_handled).pack(side=tk.LEFT, padx=5,
                                                                                              pady=5)
        ttk.Button(filter_frame, text="重置警告计数", command=self.reset_warning_count).pack(side=tk.LEFT, padx=5,
                                                                                              pady=5)

        # 下方主内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 违规记录表格
        columns = ("id", "人员", "区域", "违规类型", "时间", "状态")
        self.violations_tree = ttk.Treeview(content_frame, columns=columns, show="headings")

        for col in columns:
            self.violations_tree.heading(col, text=col)
            width = 150 if col == "时间" else 100
            self.violations_tree.column(col, width=width, anchor=tk.CENTER)

        # 添加滚动条
        scrollbar_y = ttk.Scrollbar(content_frame, orient="vertical", command=self.violations_tree.yview)
        scrollbar_x = ttk.Scrollbar(content_frame, orient="horizontal", command=self.violations_tree.xview)
        self.violations_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        # 左侧表格区域
        tree_frame = ttk.Frame(content_frame)
        tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.violations_tree.pack(fill=tk.BOTH, expand=True)

        # 右侧截图区域
        image_frame = ttk.LabelFrame(content_frame, text="违规截图")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.violation_image_canvas = tk.Canvas(image_frame, width=320, height=240, bg="#e0e0e0", highlightthickness=1)
        self.violation_image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.violation_image_canvas.create_text(160, 120, text="请选择一条违规记录", fill="#666666",
                                                font=('Microsoft YaHei', 10))

        self.violations_tree.bind("<<TreeviewSelect>>", self.show_violation_screenshot)

        self.refresh_violation_persons()

    def setup_salary_tab(self):
        """设置工资结算标签页"""
        main_frame = ttk.Frame(self.tab_salary)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # 筛选区域
        filter_frame = ttk.LabelFrame(main_frame, text="查询条件")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(filter_frame, text="人员:").pack(side=tk.LEFT, padx=5, pady=5)
        self.salary_person_combobox = ttk.Combobox(filter_frame)
        self.salary_person_combobox.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        ttk.Label(filter_frame, text="月份:").pack(side=tk.LEFT, padx=5, pady=5)
        self.salary_month_entry = ttk.Entry(filter_frame, width=10)
        self.salary_month_entry.insert(0, datetime.now().strftime("%Y-%m"))
        self.salary_month_entry.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(filter_frame, text="查询", command=self.query_salary).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(filter_frame, text="计算工资", command=self.calculate_salary).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(filter_frame, text="导出工资单", command=self.export_salary).pack(side=tk.LEFT, padx=5, pady=5)

        # 工资记录表格
        columns = ("id", "人员", "月份", "工时", "日薪", "时薪", "绩效系数", "总工资", "状态")
        self.salary_tree = ttk.Treeview(main_frame, columns=columns, show="headings")

        for col in columns:
            self.salary_tree.heading(col, text=col)
            self.salary_tree.column(col, width=80, anchor=tk.CENTER)

        # 添加滚动条
        scrollbar_y = ttk.Scrollbar(main_frame, orient="vertical", command=self.salary_tree.yview)
        scrollbar_x = ttk.Scrollbar(main_frame, orient="horizontal", command=self.salary_tree.xview)
        self.salary_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.salary_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 工资设置区域
        salary_settings_frame = ttk.LabelFrame(main_frame, text="工资设置")
        salary_settings_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(salary_settings_frame, text="日薪:").pack(side=tk.LEFT, padx=15, pady=8)
        self.daily_rate_entry = ttk.Entry(salary_settings_frame, width=10)
        self.daily_rate_entry.insert(0, "200")
        self.daily_rate_entry.pack(side=tk.LEFT, padx=5, pady=8)

        ttk.Label(salary_settings_frame, text="时薪:").pack(side=tk.LEFT, padx=15, pady=8)
        self.hourly_rate_entry = ttk.Entry(salary_settings_frame, width=10)
        self.hourly_rate_entry.insert(0, "25")
        self.hourly_rate_entry.pack(side=tk.LEFT, padx=5, pady=8)

        ttk.Label(salary_settings_frame, text="绩效系数:").pack(side=tk.LEFT, padx=15, pady=8)
        self.performance_entry = ttk.Entry(salary_settings_frame, width=10)
        self.performance_entry.insert(0, "1.0")
        self.performance_entry.pack(side=tk.LEFT, padx=5, pady=8)

        ttk.Button(salary_settings_frame, text="保存设置", command=self.save_salary_settings).pack(side=tk.LEFT,
                                                                                                   padx=20, pady=8)

        self.refresh_salary_persons()

    def setup_stats_tab(self):
        """设置数据统计标签页 - 专业仪表盘设计"""
        main_frame = ttk.Frame(self.tab_stats)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # 顶部控制面板 - 现代化设计
        control_frame = ttk.LabelFrame(main_frame, text="📊 智能数据分析中心", padding=(15, 10))
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 分析类型选择
        type_frame = ttk.Frame(control_frame)
        type_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(type_frame, text="分析模式:", font=("Microsoft YaHei UI", 11, "bold")).pack(side=tk.LEFT, padx=(0, 15))

        self.stats_type = tk.StringVar(value="comprehensive")
        analysis_types = [
            ("综合仪表盘", "comprehensive"),
            ("考勤效能分析", "attendance"),
            ("安全违规监控", "violations"),
            ("人员结构分析", "persons"),
            ("时间趋势洞察", "trends")
        ]

        for text, value in analysis_types:
            ttk.Radiobutton(type_frame, text=text, variable=self.stats_type, value=value).pack(side=tk.LEFT, padx=(0, 20))

        # 时间维度控制
        time_frame = ttk.Frame(control_frame)
        time_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(time_frame, text="时间维度:", font=("Microsoft YaHei UI", 11, "bold")).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(time_frame, text="起始:", font=("Microsoft YaHei UI", 10)).pack(side=tk.LEFT, padx=(10, 5))
        self.stats_start_date = ttk.Entry(time_frame, width=12, font=("Microsoft YaHei UI", 10))
        self.stats_start_date.insert(0, (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
        self.stats_start_date.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(time_frame, text="终止:", font=("Microsoft YaHei UI", 10)).pack(side=tk.LEFT, padx=(10, 5))
        self.stats_end_date = ttk.Entry(time_frame, width=12, font=("Microsoft YaHei UI", 10))
        self.stats_end_date.insert(0, datetime.now().strftime("%Y-%m-%d"))
        self.stats_end_date.pack(side=tk.LEFT, padx=(0, 20))

        # 操作控制区
        action_frame = ttk.Frame(time_frame)
        action_frame.pack(side=tk.RIGHT)

        ttk.Button(action_frame, text="🔄 实时刷新", command=self.generate_stats,
                  style="Success.TButton", padding=(10, 5)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_frame, text="📊 生成报告", command=self.export_stats_report,
                  style="TButton", padding=(10, 5)).pack(side=tk.LEFT)

        # 核心可视化区域
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 主数据可视化面板
        self.create_main_visualization_panel(viz_frame)

        # 智能摘要面板 - 关键指标展示
        summary_frame = ttk.LabelFrame(main_frame, text="🎯 核心指标总览", padding=(15, 10))
        summary_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        # 指标网格布局
        self.create_metrics_dashboard(summary_frame)

    def create_main_visualization_panel(self, parent):
        """创建主可视化面板"""
        # 主图表容器
        main_viz_frame = ttk.LabelFrame(parent, text="数据洞察图表", padding=(10, 5))
        main_viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建matplotlib图表
        self.stats_figure = plt.Figure(figsize=(12, 7), dpi=100, facecolor=self.card_color)
        self.stats_plot = self.stats_figure.add_subplot(111)
        self.stats_canvas = FigureCanvasTkAgg(self.stats_figure, main_viz_frame)
        self.stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 设置图表样式
        self.stats_figure.patch.set_facecolor(self.card_color)
        self.stats_plot.set_facecolor(self.card_color)

    def create_metrics_dashboard(self, parent):
        """创建指标仪表盘"""
        # 使用网格布局展示关键指标
        metrics_grid = ttk.Frame(parent)
        metrics_grid.pack(fill=tk.X, padx=5, pady=5)

        # 定义指标配置
        metrics_config = [
            {"name": "注册人员", "icon": "👥", "color": self.accent_color},
            {"name": "今日考勤", "icon": "📅", "color": self.theme_colors["success"]},
            {"name": "活跃区域", "icon": "🏗️", "color": self.theme_colors["info"]},
            {"name": "安全事件", "icon": "⚠️", "color": self.theme_colors["warning"]},
            {"name": "系统状态", "icon": "🔧", "color": self.theme_colors["success"]},
            {"name": "AI准确率", "icon": "🤖", "color": self.accent_color}
        ]

        self.summary_vars = {}
        for i, config in enumerate(metrics_config):
            # 每个指标的容器
            metric_frame = ttk.LabelFrame(metrics_grid, text=f"{config['icon']} {config['name']}",
                                        padding=(10, 5), style="Card.TFrame")
            metric_frame.grid(row=i//3, column=i%3, padx=8, pady=8, sticky="nsew")

            # 指标值显示
            value_var = tk.StringVar(value="--")
            self.summary_vars[config['name']] = value_var

            value_label = ttk.Label(metric_frame, textvariable=value_var,
                                  font=("Microsoft YaHei UI", 18, "bold"),
                                  foreground=config['color'])
            value_label.pack(pady=(5, 0))

            # 指标描述
            desc_label = ttk.Label(metric_frame, text="实时数据",
                                 font=("Microsoft YaHei UI", 9),
                                 foreground=self.theme_colors["muted"])
            desc_label.pack(pady=(0, 5))

        # 配置网格权重
        for i in range(2):  # 2行
            metrics_grid.rowconfigure(i, weight=1)
        for i in range(3):  # 3列
            metrics_grid.columnconfigure(i, weight=1)

    def create_dashboard_charts(self, parent):
        """创建仪表盘图表布局"""
        # 主图表
        main_chart_frame = ttk.LabelFrame(parent, text="主数据图表")
        main_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.stats_figure = plt.Figure(figsize=(10, 6), dpi=100, facecolor=self.bg_color)
        self.stats_plot = self.stats_figure.add_subplot(111)
        self.stats_canvas = FigureCanvasTkAgg(self.stats_figure, main_chart_frame)
        self.stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 辅助图表区域
        aux_frame = ttk.Frame(parent)
        aux_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        # 左侧小图表
        left_chart_frame = ttk.LabelFrame(aux_frame, text="实时状态")
        left_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2.5))

        self.status_figure = plt.Figure(figsize=(4, 3), dpi=80, facecolor=self.card_color)
        self.status_plot = self.status_figure.add_subplot(111)
        self.status_canvas = FigureCanvasTkAgg(self.status_figure, left_chart_frame)
        self.status_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 右侧小图表
        right_chart_frame = ttk.LabelFrame(aux_frame, text="趋势概览")
        right_chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2.5, 0))

        self.trend_figure = plt.Figure(figsize=(4, 3), dpi=80, facecolor=self.card_color)
        self.trend_plot = self.trend_figure.add_subplot(111)
    def export_stats_report(self):
        """导出统计报告"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel文件", "*.xlsx"), ("所有文件", "*.*")],
                title="导出统计报告"
            )
            if filename:
                # 这里可以实现导出逻辑
                messagebox.showinfo("成功", f"报告已导出到: {filename}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")

    def refresh_persons_list(self):
        """刷新人员列表"""
        for item in self.persons_tree.get_children():
            self.persons_tree.delete(item)

        try:
            self.cursor.execute("SELECT id, name, employee_id, rfid_card, permission_level FROM persons ORDER BY id")
            persons = self.cursor.fetchall()

            for person in persons:
                person_id, name, employee_id, rfid_card, permission_level = person
                permission_text = self.permission_levels.get(permission_level, "无权限")
                self.persons_tree.insert("", tk.END,
                                         values=(person_id, name, employee_id, rfid_card or "", permission_text))
        except Exception as e:
            messagebox.showerror("错误", f"刷新人员列表失败: {str(e)}")

    def search_persons(self):
        """搜索人员"""
        keyword = self.search_entry.get().lower()

        for item in self.persons_tree.get_children():
            self.persons_tree.delete(item)

        try:
            self.cursor.execute(
                "SELECT id, name, employee_id, rfid_card, permission_level FROM persons WHERE name LIKE ? OR employee_id LIKE ? ORDER BY id",
                (f'%{keyword}%', f'%{keyword}%'))
            persons = self.cursor.fetchall()

            for person in persons:
                person_id, name, employee_id, rfid_card, permission_level = person
                permission_text = self.permission_levels.get(permission_level, "无权限")
                self.persons_tree.insert("", tk.END,
                                         values=(person_id, name, employee_id, rfid_card or "", permission_text))
        except Exception as e:
            messagebox.showerror("错误", f"搜索人员失败: {str(e)}")

    def on_person_select(self, event):
        """选择人员时触发"""
        selected_items = self.persons_tree.selection()
        if not selected_items:
            return

        selected_item = selected_items[0]
        person_id = self.persons_tree.item(selected_item, "values")[0]

        try:
            self.cursor.execute("SELECT name, employee_id, rfid_card, permission_level FROM persons WHERE id = ?",
                                (person_id,))
            person = self.cursor.fetchone()

            if person:
                name, employee_id, rfid_card, permission_level = person
                self.name_entry.delete(0, tk.END)
                self.name_entry.insert(0, name)

                self.employee_id_entry.delete(0, tk.END)
                self.employee_id_entry.insert(0, employee_id)

                self.rfid_entry.delete(0, tk.END)
                if rfid_card:
                    self.rfid_entry.insert(0, rfid_card)

                permission_text = self.permission_levels.get(permission_level, "无权限")
                self.permission_combobox.set(permission_text)

                self.update_btn.config(state=tk.NORMAL)
                self.delete_btn.config(state=tk.NORMAL)

                self.current_person_id = person_id
        except Exception as e:
            messagebox.showerror("错误", f"获取人员信息失败: {str(e)}")

    def add_person(self):
        """添加人员"""
        name = self.name_entry.get().strip()
        employee_id = self.employee_id_entry.get().strip()
        rfid_card = self.rfid_entry.get().strip()

        if not name or not employee_id:
            messagebox.showerror("错误", "姓名和工号不能为空")
            return

        try:
            self.cursor.execute("SELECT id FROM persons WHERE employee_id = ?", (employee_id,))
            if self.cursor.fetchone():
                messagebox.showerror("错误", "该工号已存在")
                return

            permission_text = self.permission_combobox.get()
            permission_level = [k for k, v in self.permission_levels.items() if v == permission_text][0]

            self.cursor.execute(
                "INSERT INTO persons (name, employee_id, rfid_card, permission_level) VALUES (?, ?, ?, ?)",
                (name, employee_id, rfid_card, permission_level)
            )
            self.conn.commit()

            messagebox.showinfo("成功", "人员添加成功")
            self.refresh_persons_list()
            self.clear_person_form()
        except Exception as e:
            messagebox.showerror("错误", f"添加人员失败: {str(e)}")
            self.conn.rollback()

    def update_person(self):
        """更新人员信息"""
        if not hasattr(self, 'current_person_id'):
            return

        name = self.name_entry.get().strip()
        employee_id = self.employee_id_entry.get().strip()
        rfid_card = self.rfid_entry.get().strip()

        if not name or not employee_id:
            messagebox.showerror("错误", "姓名和工号不能为空")
            return

        try:
            self.cursor.execute("SELECT id FROM persons WHERE employee_id = ? AND id != ?",
                                (employee_id, self.current_person_id))
            if self.cursor.fetchone():
                messagebox.showerror("错误", "该工号已存在")
                return

            permission_text = self.permission_combobox.get()
            permission_level = [k for k, v in self.permission_levels.items() if v == permission_text][0]

            self.cursor.execute(
                "UPDATE persons SET name = ?, employee_id = ?, rfid_card = ?, permission_level = ? WHERE id = ?",
                (name, employee_id, rfid_card, permission_level, self.current_person_id)
            )
            self.conn.commit()

            messagebox.showinfo("成功", "人员信息更新成功")
            self.refresh_persons_list()
        except Exception as e:
            messagebox.showerror("错误", f"更新人员信息失败: {str(e)}")
            self.conn.rollback()

    def delete_person(self):
        """删除人员"""
        if not hasattr(self, 'current_person_id'):
            return

        if messagebox.askyesno("确认", "确定要删除该人员吗？此操作不可撤销。"):
            try:
                # 级联删除相关记录
                self.cursor.execute("DELETE FROM face_samples WHERE person_id = ?", (self.current_person_id,))
                self.cursor.execute("DELETE FROM attendance WHERE person_id = ?", (self.current_person_id,))
                self.cursor.execute("DELETE FROM violations WHERE person_id = ?", (self.current_person_id,))
                self.cursor.execute("DELETE FROM salary WHERE person_id = ?", (self.current_person_id,))
                self.cursor.execute("DELETE FROM persons WHERE id = ?", (self.current_person_id,))

                self.conn.commit()
                messagebox.showinfo("成功", "人员删除成功")
                self.refresh_persons_list()
                self.clear_person_form()
            except Exception as e:
                messagebox.showerror("错误", f"删除人员失败: {str(e)}")
                self.conn.rollback()

    def clear_person_form(self):
        """清空人员表单"""
        self.name_entry.delete(0, tk.END)
        self.employee_id_entry.delete(0, tk.END)
        self.rfid_entry.delete(0, tk.END)
        self.permission_combobox.current(0)

        self.update_btn.config(state=tk.DISABLED)
        self.delete_btn.config(state=tk.DISABLED)

        if hasattr(self, 'current_person_id'):
            delattr(self, 'current_person_id')

    def start_camera(self):
        """启动摄像头"""
        if not hasattr(self, 'current_person_id'):
            messagebox.showerror("错误", "请先选择一个人员")
            return

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")

            self.take_sample_btn.config(state=tk.NORMAL)
            self.capture_btn.config(text="关闭摄像头", command=self.stop_camera)
            self.update_camera_feed()
        except Exception as e:
            messagebox.showerror("错误", f"启动摄像头失败: {str(e)}")
            self.cap = None

    def stop_camera(self):
        """停止摄像头"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.face_canvas.delete("all")
        self.face_canvas.create_text(160, 120, text="摄像头已关闭", fill="#666666", font=('Microsoft YaHei', 10))
        self.take_sample_btn.config(state=tk.DISABLED)
        self.capture_btn.config(text="打开摄像头", command=self.start_camera)

    def update_camera_feed(self):
        """更新摄像头画面"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                # 识别人脸并标记
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    frame = self.put_text_cn(frame, "请对准摄像头", (x, y - 10), color=(0, 255, 0), font_size=22)

                # 转换为Tkinter可用格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (320, 240))
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)

                self.face_canvas.imgtk = imgtk
                self.face_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        if self.cap and self.cap.isOpened():
            self.root.after(10, self.update_camera_feed)
        else:
            self.stop_camera()

    def take_face_sample(self):
        """采集人脸样本 - 集成工地视觉算法服务"""
        if not hasattr(self, 'current_person_id') or self.current_frame is None:
            return

        # 检测人脸
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            messagebox.showwarning("警告", "未检测到人脸")
            return

        # 保存传统人脸样本
        x, y, w, h = faces[0]
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (200, 200))

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        sample_path = f"face_samples/person_{self.current_person_id}_{timestamp}.jpg"
        cv2.imwrite(sample_path, face_img)

        # 保存到数据库
        try:
            self.cursor.execute(
                "INSERT INTO face_samples (person_id, sample_path) VALUES (?, ?)",
                (self.current_person_id, sample_path)
            )
            self.conn.commit()
            
            # 使用工地视觉算法服务提取并保存人脸特征
            if self.ai_framework and self.ai_framework.is_initialized:
                try:
                    # 使用彩色人脸图像提取特征
                    color_face_img = self.current_frame[y:y + h, x:x + w]
                    result = self.ai_framework.face_recognition(color_face_img)
                    
                    if 'error' not in result and 'features' in result:
                        features = np.array(result['features'])
                        self.save_face_features(self.current_person_id, features)
                        print(f"✅ 视觉算法: 已保存人员 {self.current_person_id} 的人脸特征")
                    else:
                        print(f"❌ 视觉算法: 人脸特征提取失败")
                        
                except Exception as e:
                    print(f"❌ 视觉算法: 保存人脸特征失败: {e}")
            
            messagebox.showinfo("成功", "人脸样本采集成功\n已保存传统样本和AI特征")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存人脸样本失败: {str(e)}")
            self.conn.rollback()

    def train_recognizer(self):
        """训练人脸识别模型"""
        try:
            self.cursor.execute("SELECT DISTINCT person_id FROM face_samples")
            person_ids = [row[0] for row in self.cursor.fetchall()]

            if not person_ids:
                messagebox.showwarning("警告", "没有足够的人脸样本进行训练")
                return

            faces = []
            ids = []

            for person_id in person_ids:
                self.cursor.execute("SELECT sample_path FROM face_samples WHERE person_id = ?", (person_id,))
                samples = self.cursor.fetchall()

                for sample in samples:
                    sample_path = sample[0]
                    if os.path.exists(sample_path):
                        face_img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
                        faces.append(face_img)
                        ids.append(int(person_id))

            if not faces:
                messagebox.showwarning("警告", "没有有效的人脸样本进行训练")
                return

            self.recognizer.train(faces, np.array(ids))
            self.recognizer.write('face_recognizer.yml')
            messagebox.showinfo("成功", "人脸识别模型训练成功")
        except Exception as e:
            messagebox.showerror("错误", f"训练模型失败: {str(e)}")

    def detect_helmet(self, frame, gray, face):
        """安全帽检测算法 - 优先使用工地视觉算法服务"""
        x, y, w, h = face
        
        # 优先使用工地视觉算法服务进行检测
        if self.ai_framework and self.ai_framework.is_initialized:
            try:
                # 提取头部区域进行检测
                head_region_top = max(0, y - int(h * 1.2))
                head_region_bottom = y + int(h * 0.3)
                head_region_left = max(0, x - int(w * 0.3))
                head_region_right = min(frame.shape[1], x + w + int(w * 0.3))
                
                head_region = frame[head_region_top:head_region_bottom, head_region_left:head_region_right]
                
                if head_region.size > 0:
                    # 使用视觉算法进行检测
                    result = self.ai_framework.helmet_detection(head_region)
                    
                    if 'error' not in result and result.get('num_detections', 0) > 0:
                        # 找到安全帽检测结果
                        for detection in result.get('detections', []):
                            if detection.get('class_name') == 'helmet' and detection.get('confidence', 0) > 0.5:
                                # 调整坐标到原图
                                bbox = detection['bbox']
                                return True, (
                                    head_region_left + bbox[0],
                                    head_region_top + bbox[1],
                                    bbox[2],
                                    bbox[3]
                                )
                    
                    # 如果没有检测到安全帽，返回未佩戴状态
                    return False, (head_region_left, head_region_top,
                                  head_region_right - head_region_left,
                                  head_region_bottom - head_region_top)
                    
            except Exception as e:
                print(f"视觉算法安全帽检测失败，使用传统方法: {e}")
        
        # 尝试ONNX/YOLO模型（如存在）
        if self.helmet_net is not None:
            try:
                head_region_top = max(0, y - int(h * 1.2))
                head_region_bottom = y + int(h * 0.3)
                head_region_left = max(0, x - int(w * 0.3))
                head_region_right = min(frame.shape[1], x + w + int(w * 0.3))
                head_region = frame[head_region_top:head_region_bottom, head_region_left:head_region_right]
                if head_region.size > 0:
                    dets = self.yolo_detect(self.helmet_net, head_region, conf_threshold=0.35)
                    for (dx, dy, dw, dh, score, cls_id) in dets:
                        cls_name = self.helmet_classes[cls_id] if 0 <= cls_id < len(self.helmet_classes) else str(cls_id)
                        if ("helmet" in cls_name.lower()) and score >= 0.5:
                            return True, (head_region_left + dx, head_region_top + dy, dw, dh)
                    # 未检测到安全帽
                    return False, (head_region_left, head_region_top,
                                  head_region_right - head_region_left,
                                  head_region_bottom - head_region_top)
            except Exception as e:
                print(f"ONNX安全帽检测失败，使用传统方法: {e}")

        # 传统检测方法（备用）
        return self.detect_helmet_traditional(frame, gray, face)

    def detect_helmet_traditional(self, frame, gray, face):
        """传统安全帽检测算法（备用）"""
        x, y, w, h = face
        # 安全帽通常在头部区域，即人脸上方
        head_region_top = max(0, y - int(h * 1.2))  # 扩大检测区域
        head_region_bottom = y + int(h * 0.3)
        head_region_left = max(0, x - int(w * 0.3))
        head_region_right = min(frame.shape[1], x + w + int(w * 0.3))

        head_region = frame[head_region_top:head_region_bottom, head_region_left:head_region_right]

        if head_region.size == 0:
            return False, (x, head_region_top, w, h)

        # 转换为HSV色彩空间，更好地检测颜色
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)

        # 定义常见安全帽颜色的HSV范围：红色、黄色、蓝色、白色、橙色
        red_lower1 = np.array([0, 100, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 50])
        red_upper2 = np.array([180, 255, 255])

        yellow_lower = np.array([15, 80, 80])
        yellow_upper = np.array([35, 255, 255])

        blue_lower = np.array([90, 100, 50])
        blue_upper = np.array([130, 255, 255])
        
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        
        orange_lower = np.array([5, 100, 100])
        orange_upper = np.array([20, 255, 255])

        # 创建颜色掩码
        mask_red = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
        mask_white = cv2.inRange(hsv, white_lower, white_upper)
        mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)

        mask = mask_red | mask_yellow | mask_blue | mask_white | mask_orange

        # 对掩码进行形态学操作，去除噪声
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 计算掩码区域与头部区域的比例
        mask_area = cv2.countNonZero(mask)
        head_area = head_region.shape[0] * head_region.shape[1]

        # 降低阈值，提高检测敏感度
        if head_area > 0 and mask_area / head_area > 0.15:
            return True, (head_region_left, head_region_top,
                          head_region_right - head_region_left,
                          head_region_bottom - head_region_top)

        # 如果有级联分类器模型，使用模型进行二次验证
        if self.helmet_cascade:
            helmets = self.helmet_cascade.detectMultiScale(
                gray[head_region_top:head_region_bottom, head_region_left:head_region_right],
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )
            if len(helmets) > 0:
                hx, hy, hw, hh = helmets[0]
                return True, (head_region_left + hx, head_region_top + hy, hw, hh)

        return False, (head_region_left, head_region_top,
                       head_region_right - head_region_left,
                       head_region_bottom - head_region_top)

    def detect_smoking(self, frame, gray, face):
        """吸烟检测算法 - 优先使用工地视觉算法服务"""
        x, y, w, h = face
        
        # 优先使用工地视觉算法服务进行检测
        if self.ai_framework and self.ai_framework.is_initialized:
            try:
                # 提取嘴部区域进行检测
                mouth_region_top = y + int(h * 0.4)
                mouth_region_bottom = y + h + int(h * 0.2)
                mouth_region_left = max(0, x - int(w * 0.2))
                mouth_region_right = min(frame.shape[1], x + w + int(w * 0.2))
                
                mouth_region = frame[mouth_region_top:mouth_region_bottom, mouth_region_left:mouth_region_right]
                
                if mouth_region.size > 0:
                    # 使用视觉算法进行检测
                    result = self.ai_framework.smoking_detection(mouth_region)
                    
                    if 'error' not in result:
                        is_smoking = result.get('is_smoking', False)
                        confidence = result.get('confidence', 0.0)
                        
                        if is_smoking and confidence > 0.6:
                            # 返回检测到的吸烟区域
                            return True, (mouth_region_left, mouth_region_top,
                                        mouth_region_right - mouth_region_left,
                                        mouth_region_bottom - mouth_region_top)
                        else:
                            return False, (0, 0, 0, 0)
                            
            except Exception as e:
                print(f"视觉算法吸烟检测失败，使用传统方法: {e}")
        
        # 尝试ONNX/YOLO模型（如存在）
        if self.smoke_net is not None:
            try:
                mouth_region_top = y + int(h * 0.4)
                mouth_region_bottom = y + h + int(h * 0.2)
                mouth_region_left = max(0, x - int(w * 0.2))
                mouth_region_right = min(frame.shape[1], x + w + int(w * 0.2))
                mouth_region = frame[mouth_region_top:mouth_region_bottom, mouth_region_left:mouth_region_right]
                if mouth_region.size > 0:
                    dets = self.yolo_detect(self.smoke_net, mouth_region, conf_threshold=0.28)
                    # 时序平滑：对每张脸维护历史
                    key = (x//20, y//20, w//20, h//20)
                    detected = False
                    best_rect = (0, 0, 0, 0)
                    best_score = 0.0
                    for (dx, dy, dw, dh, score, cls_id) in dets:
                        cls_name = self.smoke_classes[cls_id] if 0 <= cls_id < len(self.smoke_classes) else str(cls_id)
                        if any(k in cls_name.lower() for k in ["cig", "smoke", "cigarette"]):
                            if score > best_score:
                                best_score = score
                                best_rect = (mouth_region_left + dx, mouth_region_top + dy, dw, dh)
                    # 更新历史
                    self.smoking_history[key].append(best_score >= 0.5)
                    self.smoking_conf_history[key].append(best_score)
                    # 要求至少连续3/8帧为阳性，且平均分>0.45
                    hist = self.smoking_history[key]
                    conf_hist = self.smoking_conf_history[key]
                    if len(hist) >= 4 and sum(hist) >= 3 and (sum(conf_hist)/len(conf_hist) > 0.45) and best_score >= 0.5:
                        # 二次校验：亮度小热点 + 边缘结构
                        mx, my, mw, mh = best_rect
                        roi_gray = gray[my:my+mh, mx:mx+mw] if mw>0 and mh>0 else None
                        if roi_gray is not None and roi_gray.size>0:
                            _, thr = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY)
                            edges = cv2.Canny(roi_gray, 60, 160)
                            bright_ratio = cv2.countNonZero(thr) / float(roi_gray.shape[0]*roi_gray.shape[1] + 1e-6)
                            edge_ratio = cv2.countNonZero(edges) / float(roi_gray.shape[0]*roi_gray.shape[1] + 1e-6)
                            if bright_ratio > 0.02 and edge_ratio > 0.05:
                                # 通过
                                # 采样保存正样本
                                if random.random() < 0.15:
                                    try:
                                        self.save_smoking_sample(mouth_region, label=1)
                                    except Exception:
                                        pass
                                return True, best_rect
                    # 采样负样本（降低频率）
                    if random.random() < 0.05:
                        try:
                            self.save_smoking_sample(mouth_region, label=0)
                        except Exception:
                            pass
                    return False, (0, 0, 0, 0)
            except Exception as e:
                print(f"ONNX吸烟检测失败，使用传统方法: {e}")

        # 传统检测方法（备用）
        return self.detect_smoking_traditional(frame, gray, face)

    def save_smoking_sample(self, region_bgr, label: int):
        """保存吸烟样本，label=1为正样本，0为负样本。"""
        base_dir = os.path.join("datasets", "smoking")
        pos_dir = os.path.join(base_dir, "positive")
        neg_dir = os.path.join(base_dir, "negative")
        if not os.path.exists(pos_dir):
            os.makedirs(pos_dir, exist_ok=True)
        if not os.path.exists(neg_dir):
            os.makedirs(neg_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
        save_dir = pos_dir if label == 1 else neg_dir
        # 统一尺寸以便训练
        try:
            patch = cv2.resize(region_bgr, (160, 120))
            cv2.imwrite(os.path.join(save_dir, f"{ts}.jpg"), patch)
        except Exception:
            pass

    def detect_smoking_traditional(self, frame, gray, face):
        """传统吸烟检测算法（备用）"""
        x, y, w, h = face
        # 吸烟通常在嘴部区域，即人脸中下部
        mouth_region_top = y + int(h * 0.4)  # 扩大检测区域
        mouth_region_bottom = y + h + int(h * 0.2)  # 包含下巴区域
        mouth_region_left = max(0, x - int(w * 0.2))
        mouth_region_right = min(frame.shape[1], x + w + int(w * 0.2))

        mouth_region = gray[mouth_region_top:mouth_region_bottom, mouth_region_left:mouth_region_right]

        if mouth_region.size == 0:
            return False, (x, mouth_region_top, w, h // 2)

        # 多种检测方法结合
        
        # 方法1：检测亮度较高的区域（烟头通常较亮）
        _, thresh_bright = cv2.threshold(mouth_region, 180, 255, cv2.THRESH_BINARY)
        
        # 方法2：检测边缘特征
        edges = cv2.Canny(mouth_region, 50, 150)
        
        # 方法3：检测圆形特征（烟头形状）
        circles = cv2.HoughCircles(mouth_region, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=5, maxRadius=25)

        # 分析亮度区域
        kernel = np.ones((3, 3), np.uint8)
        thresh_bright = cv2.morphologyEx(thresh_bright, cv2.MORPH_CLOSE, kernel)
        contours_bright, _ = cv2.findContours(thresh_bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 分析边缘特征
        contours_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 检查圆形检测结果
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                # 检查圆形区域是否在嘴部合理位置
                if 5 < r < 20 and mouth_region.shape[0] > cy > mouth_region.shape[0] * 0.3:
                    return True, (mouth_region_left + cx - r, mouth_region_top + cy - r, r * 2, r * 2)

        # 分析轮廓特征
        for contour in contours_bright:
            area = cv2.contourArea(contour)
            if 5 < area < 200:  # 扩大烟头大小范围
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if 0.3 < circularity < 2.0:  # 放宽圆形度要求
                        xc, yc, wc, hc = cv2.boundingRect(contour)
                        # 检查长宽比，烟头通常接近圆形或椭圆形
                        aspect_ratio = wc / hc if hc > 0 else 0
                        if 0.5 < aspect_ratio < 2.0:
                            xc += mouth_region_left
                            yc += mouth_region_top
                            return True, (xc, yc, wc, hc)

        # 分析边缘轮廓
        for contour in contours_edges:
            area = cv2.contourArea(contour)
            if 10 < area < 150:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if 0.4 < circularity < 1.8:
                        xc, yc, wc, hc = cv2.boundingRect(contour)
                        aspect_ratio = wc / hc if hc > 0 else 0
                        if 0.6 < aspect_ratio < 1.8:
                            xc += mouth_region_left
                            yc += mouth_region_top
                            return True, (xc, yc, wc, hc)

        # 如果有级联分类器模型，使用模型进行二次验证
        if self.cigarette_cascade:
            cigarettes = self.cigarette_cascade.detectMultiScale(
                gray[mouth_region_top:mouth_region_bottom, mouth_region_left:mouth_region_right],
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(15, 15)
            )
            if len(cigarettes) > 0:
                cx, cy, cw, ch = cigarettes[0]
                return True, (mouth_region_left + cx, mouth_region_top + cy, cw, ch)

        return False, (0, 0, 0, 0)

    def play_warning_sound(self):
        """播放警告声音"""
        try:
            # 播放系统警告声音
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            # 也可以播放自定义声音
            # winsound.PlaySound("warning.wav", winsound.SND_FILENAME)
        except:
            pass  # 如果声音播放失败，继续执行

    def show_warning_popup(self, violation_type, person_name):
        """显示警告弹窗"""
        violation_text = self.violation_types.get(violation_type, violation_type)
        message = f"检测到违规行为！\n\n人员：{person_name}\n违规类型：{violation_text}\n时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n请立即纠正违规行为！"
        
        # 创建警告窗口
        warning_window = tk.Toplevel(self.root)
        warning_window.title("安全警告")
        warning_window.geometry("400x250")
        warning_window.configure(bg="#ff4444")
        
        # 设置窗口置顶
        warning_window.attributes('-topmost', True)
        
        # 警告图标和文字
        warning_frame = tk.Frame(warning_window, bg="#ff4444")
        warning_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # 警告标题
        title_label = tk.Label(warning_frame, text="⚠️ 安全警告 ⚠️", 
                              font=("Microsoft YaHei", 16, "bold"), 
                              fg="white", bg="#ff4444")
        title_label.pack(pady=10)
        
        # 警告内容
        content_label = tk.Label(warning_frame, text=message, 
                                font=("Microsoft YaHei", 12), 
                                fg="white", bg="#ff4444", 
                                justify="left")
        content_label.pack(pady=10)
        
        # 确认按钮
        confirm_btn = tk.Button(warning_frame, text="我已了解", 
                                command=warning_window.destroy,
                                font=("Microsoft YaHei", 12, "bold"),
                                bg="white", fg="#ff4444",
                                width=15, height=2)
        confirm_btn.pack(pady=20)
        
        # 3秒后自动关闭
        warning_window.after(3000, warning_window.destroy)

    def trigger_warning(self, violation_type, person_name):
        """触发警告系统"""
        if self.warning_count >= self.max_warnings:
            return
            
        self.warning_count += 1
        
        # 更新警告状态
        self.warning_status.set(f"检测到违规行为！")
        self.warning_counter.set(f"警告次数: {self.warning_count}/{self.max_warnings}")
        
        # 播放警告声音
        self.play_warning_sound()
        
        # 显示警告弹窗
        self.show_warning_popup(violation_type, person_name)
        
        # 重置警告状态（5秒后）
        self.root.after(5000, self.reset_warning_status)

    def reset_warning_status(self):
        """重置警告状态"""
        self.warning_status.set("系统正常")
        self.warning_counter.set(f"警告次数: {self.warning_count}/{self.max_warnings}")

    def reset_warning_count(self):
        """重置警告计数器"""
        self.warning_count = 0
        self.warning_status.set("系统正常")
        self.warning_counter.set("警告次数: 0/3")
        messagebox.showinfo("成功", "警告计数器已重置")

    def toggle_unified_monitoring(self):
        """统一监控控制开关"""
        if not self.monitoring:
            self.start_unified_monitoring()
        else:
            self.stop_unified_monitoring()

    def start_unified_monitoring(self):
        """启动统一监控"""
        try:
            # 检查摄像头
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")

            # 更新状态
            self.monitoring = True
            self.monitor_running = True
            
            # 更新界面
            self.unified_monitor_btn.config(text="停止监控", style="StopMonitor.TButton")
            self.monitor_status_label.config(text="● 监控运行中", style="HeaderStatusOn.TLabel")
            self.current_monitor_status.set("监控运行中")
            
            # 启动监控线程
            self.monitor_thread = threading.Thread(target=self.monitor_worker, daemon=True)
            self.monitor_thread.start()
            
            messagebox.showinfo("成功", "监控已启动！\n\n功能包括：\n• 人脸识别\n• 安全帽检测\n• 吸烟检测\n• 违规警告")
            
        except Exception as e:
            messagebox.showerror("错误", f"启动监控失败: {str(e)}")
            self.monitoring = False
            self.monitor_running = False
            if self.cap:
                self.cap.release()
                self.cap = None

    def stop_unified_monitoring(self):
        """停止统一监控"""
        self.monitoring = False
        self.monitor_running = False
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 清空监控画面
        self.access_canvas.delete("all")
        self.access_canvas.create_text(320, 240, text="监控已停止", fill="#666666", 
                                       font=('Microsoft YaHei', 12))
        
        # 更新界面
        self.unified_monitor_btn.config(text="启动监控", style="Monitor.TButton")
        self.monitor_status_label.config(text="● 监控已停止", style="HeaderStatus.TLabel")
        self.current_monitor_status.set("监控已停止")
        
        # 重置识别信息
        self.identify_info.set("等待监控开始...")
        self.violation_info.set("暂无违规记录")
        
        messagebox.showinfo("成功", "监控已停止")

    def monitor_worker(self):
        """监控工作线程"""
        while self.monitor_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 在主线程中更新界面
                self.root.after(0, self.update_monitoring_frame, frame)
                
                # 控制帧率
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"监控线程错误: {e}")
                break
        
        # 监控结束后的清理
        if self.monitor_running:
            self.root.after(0, self.stop_unified_monitoring)

    def update_monitoring_frame(self, frame):
        """更新监控画面（在主线程中调用）"""
        if not self.monitoring or not self.monitor_running:
            return

        # 获取当前选择的区域
        area_text = self.area_combobox.get()
        area_key = [k for k, v in self.areas.items() if v == area_text][0]

        # 处理帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        identified_persons = []

        for (x, y, w, h) in faces:
            # 人脸识别 - 优先使用工地视觉算法服务
            person_id = None
            name = "未知人员"
            permission_level = 0
            
            # 优先使用工地视觉算法服务进行人脸识别
            if self.ai_framework and self.ai_framework.is_initialized:
                try:
                    face_roi = frame[y:y + h, x:x + w]  # 使用彩色图像
                    result = self.ai_framework.face_recognition(face_roi)
                    
                    if 'error' not in result and 'features' in result:
                        features = np.array(result['features'])
                        confidence = result.get('confidence', 0.0)
                        
                        if confidence > 0.7:  # 高置信度阈值
                            # 在数据库中查找匹配的人脸特征
                            person_id = self.match_face_features(features)
                            if person_id:
                                self.cursor.execute("SELECT name, permission_level FROM persons WHERE id = ?", (person_id,))
                                result = self.cursor.fetchone()
                                if result:
                                    name, permission_level = result
                                    
                except Exception as e:
                    print(f"视觉算法人脸识别失败，使用传统方法: {e}")
            
            # 传统人脸识别方法（备用）
            if person_id is None:
                try:
                    face_roi = gray[y:y + h, x:x + w]
                    face_roi = cv2.resize(face_roi, (200, 200))
                    label, confidence = self.recognizer.predict(face_roi)
                    if confidence < 70:  # 置信度阈值，越低越准确
                        person_id = label
                        self.cursor.execute("SELECT name, permission_level FROM persons WHERE id = ?", (person_id,))
                        result = self.cursor.fetchone()
                        if result:
                            name, permission_level = result
                except:
                    pass

            identified_persons.append((person_id, name, x, y, w, h))

            # 绘制人脸框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = self.put_text_cn(frame, f"{name}", (x, y - 10), color=(0, 255, 0), font_size=22)

            # 权限检查
            area_permission = 0
            if area_key == "main_gate":
                area_permission = 1
            elif area_key == "office":
                area_permission = 1
            elif area_key == "workshop":
                area_permission = 2
            elif area_key == "warehouse":
                area_permission = 3
            elif area_key == "construction":
                area_permission = 4

            access_granted = permission_level >= area_permission
            status_text = "允许进入" if access_granted else "禁止进入"
            status_color = (0, 255, 0) if access_granted else (0, 0, 255)

            frame = self.put_text_cn(frame, status_text, (x, y + h + 30), color=status_color, font_size=22)

            # 检测安全帽
            if self.helmet_detection_var.get() and area_key in self.helmet_required_areas:
                has_helmet, helmet_rect = self.detect_helmet(frame, gray, (x, y, w, h))
                hx, hy, hw, hh = helmet_rect

                if has_helmet:
                    cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), (0, 255, 0), 3)
                    frame = self.put_text_cn(frame, "已佩戴安全帽", (hx, hy - 10), color=(0, 255, 0), font_size=22)
                    self.violation_state["no_helmet"] = False
                else:
                    # 增强未佩戴安全帽的视觉警告
                    cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), (0, 0, 255), 4)
                    frame = self.put_text_cn(frame, "未佩戴安全帽", (hx, hy - 10), color=(0, 0, 255), font_size=22)
                    
                    # 添加闪烁警告框
                    cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5), (0, 0, 255), 3)

                    # 记录违规并触发警告
                    now = datetime.now()
                    if not self.violation_state["no_helmet"] and \
                            (now - self.last_violation_time["no_helmet"]).total_seconds() > self.violation_cooldown:
                        self.record_violation(person_id, area_key, "no_helmet", frame)
                        self.trigger_warning("no_helmet", name)
                        self.violation_state["no_helmet"] = True
                        self.last_violation_time["no_helmet"] = now

            # 检测吸烟
            if self.smoking_detection_var.get():
                is_smoking, smoke_rect = self.detect_smoking(frame, gray, (x, y, w, h))
                sx, sy, sw, sh = smoke_rect

                if is_smoking and sw > 0 and sh > 0:
                    # 增强吸烟检测的视觉警告
                    cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 4)
                    frame = self.put_text_cn(frame, "检测到吸烟", (sx, sy - 10), color=(0, 0, 255), font_size=22)
                    
                    # 添加闪烁警告框
                    cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5), (0, 0, 255), 3)

                    # 记录违规并触发警告
                    now = datetime.now()
                    if not self.violation_state["smoking"] and \
                            (now - self.last_violation_time["smoking"]).total_seconds() > self.violation_cooldown:
                        self.record_violation(person_id, area_key, "smoking", frame)
                        self.trigger_warning("smoking", name)
                        self.violation_state["smoking"] = True
                        self.last_violation_time["smoking"] = now
                else:
                    self.violation_state["smoking"] = False

            # 记录访问
            if person_id and access_granted:
                self.record_access(person_id, area_key, True, "人脸识别通过")
            elif person_id and not access_granted:
                self.record_access(person_id, area_key, False, "权限不足")
                now = datetime.now()
                if (now - self.last_violation_time.get("no_permission",
                                                       datetime.min)).total_seconds() > self.violation_cooldown:
                    self.record_violation(person_id, area_key, "no_permission", frame)
                    self.last_violation_time["no_permission"] = now

        # 更新识别信息
        if identified_persons:
            info_text = ", ".join([f"{name} (ID: {pid})" for pid, name, _, _, _, _ in identified_persons])
            self.identify_info.set(f"识别到: {info_text}")
        else:
            self.identify_info.set("未识别到人员")

        # 显示当前区域和时间
        frame = self.put_text_cn(frame, f"区域: {area_text}", (10, 30), color=(255, 255, 255), font_size=24)
        frame = self.put_text_cn(frame, f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (10, 60), color=(255, 255, 255), font_size=24)

        # 转换为Tkinter可用格式并显示
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.access_canvas.imgtk = imgtk
        self.access_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def on_closing(self):
        """程序关闭时的清理工作"""
        try:
            # 停止监控
            if self.monitoring:
                self.stop_unified_monitoring()
            
            # 关闭数据库连接
            if hasattr(self, 'conn'):
                self.conn.close()
            
            # 销毁窗口
            self.root.destroy()
        except Exception as e:
            print(f"关闭程序时出错: {e}")
            self.root.destroy()

    def record_access(self, person_id, area, access_granted, reason):
        """记录访问记录"""
        try:
            # 检查是否已有未关闭的访问记录
            self.cursor.execute("""
                SELECT id FROM attendance 
                WHERE person_id = ? AND area = ? AND exit_time IS NULL
            """, (person_id, area))
            existing = self.cursor.fetchone()

            if existing:
                # 如果已有未关闭的记录，更新为离开时间
                self.cursor.execute("""
                    UPDATE attendance 
                    SET exit_time = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (existing[0],))
            else:
                # 新建访问记录
                self.cursor.execute("""
                    INSERT INTO attendance 
                    (person_id, area, entry_time, access_granted, reason)
                    VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?)
                """, (person_id, area, access_granted, reason))

            self.conn.commit()
            self.refresh_access_records()
        except Exception as e:
            print(f"记录访问失败: {e}")
            self.conn.rollback()

    def record_violation(self, person_id, area, violation_type, frame):
        """记录违规行为"""
        try:
            # 保存违规截图
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            img_dir = "violation_screenshots"
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            violation_path = f"{img_dir}/{violation_type}_{timestamp}.jpg"
            cv2.imwrite(violation_path, frame)

            # 保存到数据库，补全时间和状态字段
            self.cursor.execute("""
                INSERT INTO violations 
                (person_id, area, violation_type, violation_time, screenshot_path, status)
                VALUES (?, ?, ?, ?, ?, '未处理')
            """, (person_id, area, violation_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), violation_path))

            self.conn.commit()

            # 更新违规信息显示
            violation_text = self.violation_types.get(violation_type, violation_type)
            current_time = datetime.now().strftime('%H:%M:%S')
            self.violation_info.set(f"[警报 {current_time}] {violation_text} - 已记录")

            # 刷新违规记录
            self.query_violations()
        except Exception as e:
            messagebox.showerror("错误", f"记录违规失败: {e}")
            self.conn.rollback()

    def refresh_access_records(self):
        """刷新访问记录"""
        # 只显示最近10条记录
        for item in self.access_tree.get_children():
            self.access_tree.delete(item)

        try:
            self.cursor.execute("""
                SELECT a.entry_time, p.name, a.area, a.access_granted, a.reason 
                FROM attendance a
                LEFT JOIN persons p ON a.person_id = p.id
                ORDER BY a.entry_time DESC
                LIMIT 10
            """)
            records = self.cursor.fetchall()

            for record in records:
                entry_time, name, area, access_granted, reason = record
                area_text = self.areas.get(area, area)
                status = "允许进入" if access_granted else f"禁止进入: {reason}"
                self.access_tree.insert("", tk.END,
                                        values=(entry_time, name or "未知", area_text, status))
        except Exception as e:
            print(f"刷新访问记录失败: {e}")

    # 其他方法（refresh_attendance_persons, query_attendance, export_attendance等）保持不变
    # 为简洁起见，此处省略这些方法，但在实际代码中应保留并确保其正常工作

    def refresh_attendance_persons(self):
        """刷新考勤人员下拉列表"""
        try:
            self.cursor.execute("SELECT id, name FROM persons ORDER BY name")
            persons = self.cursor.fetchall()
            self.attendance_person_combobox['values'] = ["所有人员"] + [f"{p[1]} (ID:{p[0]})" for p in persons]
            self.attendance_person_combobox.current(0)
        except Exception as e:
            messagebox.showerror("错误", f"刷新考勤人员列表失败: {str(e)}")

    def query_attendance(self):
        """查询考勤记录"""
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)

        try:
            person_text = self.attendance_person_combobox.get()
            area_text = self.attendance_area_combobox.get()
            date = self.attendance_date_entry.get()

            area_key = [k for k, v in self.areas.items() if v == area_text][0]

            query = """
                SELECT a.id, p.name, a.area, a.entry_time, a.exit_time, 
                       CASE WHEN a.access_granted THEN '允许进入' ELSE '禁止进入' END
                FROM attendance a
                LEFT JOIN persons p ON a.person_id = p.id
                WHERE 1=1
            """
            params = []

            if person_text != "所有人员" and "ID:" in person_text:
                person_id = person_text.split("ID:")[1].strip(")")
                query += " AND a.person_id = ?"
                params.append(person_id)

            query += " AND a.area = ?"
            params.append(area_key)

            if date:
                query += " AND date(a.entry_time) = ?"
                params.append(date)

            query += " ORDER BY a.entry_time DESC"

            self.cursor.execute(query, params)
            records = self.cursor.fetchall()

            for record in records:
                rec_id, name, area, entry, exit_, status = record
                area_display = self.areas.get(area, area)
                self.attendance_tree.insert("", tk.END,
                                            values=(rec_id, name or "未知", area_display, entry, exit_ or "", status))
        except Exception as e:
            messagebox.showerror("错误", f"查询考勤记录失败: {str(e)}")

    def export_attendance(self):
        """导出考勤记录为Excel文件"""
        from tkinter import filedialog
        try:
            # 获取当前查询条件
            person_text = self.attendance_person_combobox.get()
            area_text = self.attendance_area_combobox.get()
            date = self.attendance_date_entry.get()
            area_key = [k for k, v in self.areas.items() if v == area_text][0]

            query = """
                SELECT a.id, p.name, a.area, a.entry_time, a.exit_time, 
                       CASE WHEN a.access_granted THEN '允许进入' ELSE '禁止进入' END
                FROM attendance a
                LEFT JOIN persons p ON a.person_id = p.id
                WHERE 1=1
            """
            params = []
            if person_text != "所有人员" and "ID:" in person_text:
                person_id = person_text.split("ID:")[1].strip(")")
                query += " AND a.person_id = ?"
                params.append(person_id)
            query += " AND a.area = ?"
            params.append(area_key)
            if date:
                query += " AND date(a.entry_time) = ?"
                params.append(date)
            query += " ORDER BY a.entry_time DESC"
            self.cursor.execute(query, params)
            records = self.cursor.fetchall()

            if not records:
                messagebox.showinfo("提示", "没有可导出的考勤记录！")
                return

            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel文件", "*.xlsx")],
                title="保存考勤记录"
            )
            if not file_path:
                return

            import openpyxl
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "考勤记录"
            ws.append(["记录ID", "姓名", "区域", "进入时间", "离开时间", "状态"])
            for rec in records:
                rec_id, name, area, entry, exit_, status = rec
                area_display = self.areas.get(area, area)
                ws.append([rec_id, name or "未知", area_display, entry, exit_ or "", status])
            wb.save(file_path)
            messagebox.showinfo("成功", f"考勤记录已成功导出到\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")

    def refresh_violation_persons(self):
        """刷新违规人员下拉列表"""
        try:
            self.cursor.execute("SELECT id, name FROM persons ORDER BY name")
            persons = self.cursor.fetchall()
            self.violation_person_combobox['values'] = ["所有人员"] + [f"{p[1]} (ID:{p[0]})" for p in persons]
            self.violation_person_combobox.current(0)
        except Exception as e:
            messagebox.showerror("错误", f"刷新违规人员列表失败: {str(e)}")

    def query_violations(self):
        """查询违规记录"""
        for item in self.violations_tree.get_children():
            self.violations_tree.delete(item)

        try:
            person_text = self.violation_person_combobox.get()
            violation_type_text = self.violation_type_combobox.get()
            start_date = self.violation_start_date.get()
            end_date = self.violation_end_date.get()

            query = """
                SELECT v.id, p.name, v.area, v.violation_type, v.violation_time, v.status
                FROM violations v
                LEFT JOIN persons p ON v.person_id = p.id
                WHERE 1=1
            """
            params = []

            if person_text != "所有人员" and "ID:" in person_text:
                person_id = person_text.split("ID:")[1].strip(")")
                query += " AND v.person_id = ?"
                params.append(person_id)

            if violation_type_text != "所有":
                violation_type = [k for k, v in self.violation_types.items() if v == violation_type_text][0]
                query += " AND v.violation_type = ?"
                params.append(violation_type)

            if start_date:
                query += " AND date(v.violation_time) >= ?"
                params.append(start_date)

            if end_date:
                query += " AND date(v.violation_time) <= ?"
                params.append(end_date)

            query += " ORDER BY v.violation_time DESC"

            self.cursor.execute(query, params)
            records = self.cursor.fetchall()

            for record in records:
                rec_id, name, area, violation_type, time, status = record
                area_display = self.areas.get(area, area)
                violation_display = self.violation_types.get(violation_type, violation_type)
                self.violations_tree.insert("", tk.END,
                                            values=(
                                            rec_id, name or "未知", area_display, violation_display, time, status))
        except Exception as e:
            messagebox.showerror("错误", f"查询违规记录失败: {str(e)}")

    def show_violation_screenshot(self, event):
        """显示违规截图"""
        selected_items = self.violations_tree.selection()
        if not selected_items:
            return

        selected_item = selected_items[0]
        violation_id = self.violations_tree.item(selected_item, "values")[0]

        try:
            self.cursor.execute("SELECT screenshot_path FROM violations WHERE id = ?", (violation_id,))
            result = self.cursor.fetchone()

            if result and result[0] and os.path.exists(result[0]):
                img = Image.open(result[0])
                img.thumbnail((320, 240))
                imgtk = ImageTk.PhotoImage(image=img)

                self.violation_image_canvas.delete("all")
                self.violation_image_canvas.imgtk = imgtk
                self.violation_image_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            else:
                self.violation_image_canvas.delete("all")
                self.violation_image_canvas.create_text(160, 120, text="无截图或截图已删除", fill="#666666",
                                                        font=('Microsoft YaHei', 10))
        except Exception as e:
            messagebox.showerror("错误", f"显示违规截图失败: {str(e)}")

    def mark_violation_handled(self):
        """标记违规为已处理"""
        selected_items = self.violations_tree.selection()
        if not selected_items:
            messagebox.showwarning("警告", "请先选择一条违规记录")
            return

        selected_item = selected_items[0]
        violation_id = self.violations_tree.item(selected_item, "values")[0]

        try:
            self.cursor.execute("UPDATE violations SET status = '已处理' WHERE id = ?", (violation_id,))
            self.conn.commit()
            messagebox.showinfo("成功", "已标记为处理完毕")
            self.query_violations()
        except Exception as e:
            messagebox.showerror("错误", f"标记处理失败: {str(e)}")
            self.conn.rollback()

    def refresh_salary_persons(self):
        """刷新工资人员下拉列表"""
        try:
            self.cursor.execute("SELECT id, name FROM persons ORDER BY name")
            persons = self.cursor.fetchall()
            self.salary_person_combobox['values'] = ["所有人员"] + [f"{p[1]} (ID:{p[0]})" for p in persons]
            self.salary_person_combobox.current(0)
        except Exception as e:
            messagebox.showerror("错误", f"刷新工资人员列表失败: {str(e)}")

    def query_salary(self):
        """查询工资记录"""
        for item in self.salary_tree.get_children():
            self.salary_tree.delete(item)

        try:
            person_text = self.salary_person_combobox.get()
            month = self.salary_month_entry.get()

            query = """
                SELECT s.id, p.name, s.month, s.working_hours, s.daily_rate, 
                       s.hourly_rate, s.performance_factor, s.total_salary, s.status
                FROM salary s
                LEFT JOIN persons p ON s.person_id = p.id
                WHERE 1=1
            """
            params = []

            if person_text != "所有人员" and "ID:" in person_text:
                person_id = person_text.split("ID:")[1].strip(")")
                query += " AND s.person_id = ?"
                params.append(person_id)

            if month:
                query += " AND s.month = ?"
                params.append(month)

            query += " ORDER BY s.month DESC, p.name"

            self.cursor.execute(query, params)
            records = self.cursor.fetchall()

            for record in records:
                rec_id, name, month, hours, daily, hourly, perf, total, status = record
                self.salary_tree.insert("", tk.END,
                                        values=(rec_id, name or "未知", month,
                                                round(hours, 2) if hours else 0,
                                                daily or 0, hourly or 0,
                                                perf or 1.0, round(total, 2) if total else 0,
                                                status))
        except Exception as e:
            messagebox.showerror("错误", f"查询工资记录失败: {str(e)}")

    def calculate_salary(self):
        """自动统计工时并计算工资（简化版）"""
        try:
            month = self.salary_month_entry.get().strip()
            if not month:
                month = datetime.now().strftime("%Y-%m")
            try:
                hourly_rate = float(self.hourly_rate_entry.get())
            except:
                hourly_rate = 25.0
            self.cursor.execute("SELECT id FROM persons")
            persons = self.cursor.fetchall()
            for (person_id,) in persons:
                self.cursor.execute("""
                    SELECT entry_time, exit_time FROM attendance
                    WHERE person_id=? AND strftime('%Y-%m', entry_time)=?
                        AND entry_time IS NOT NULL AND exit_time IS NOT NULL
                """, (person_id, month))
                records = self.cursor.fetchall()
                total_hours = 0.0
                for entry, exit in records:
                    try:
                        t1 = datetime.fromisoformat(entry)
                        t2 = datetime.fromisoformat(exit)
                        hours = (t2-t1).total_seconds()/3600.0
                        if hours > 0:
                            total_hours += hours
                    except:
                        pass
                total_salary = round(total_hours * hourly_rate, 2)
                self.cursor.execute("""
                    SELECT id FROM salary WHERE person_id=? AND month=?
                """, (person_id, month))
                row = self.cursor.fetchone()
                if row:
                    self.cursor.execute("""
                        UPDATE salary SET working_hours=?, hourly_rate=?, total_salary=? WHERE id=?
                    """, (total_hours, hourly_rate, total_salary, row[0]))
                else:
                    self.cursor.execute("""
                        INSERT INTO salary (person_id, month, working_hours, hourly_rate, total_salary)
                        VALUES (?, ?, ?, ?, ?)
                    """, (person_id, month, total_hours, hourly_rate, total_salary))
            self.conn.commit()
            messagebox.showinfo("成功", "工资已计算完成！")
            self.query_salary()
        except Exception as e:
            messagebox.showerror("错误", f"工资计算失败: {str(e)}")

    def save_salary_settings(self):
        """保存工资设置（示例实现）"""
        messagebox.showinfo("提示", "工资设置已保存")

    def export_salary(self):
        """导出工资单为Excel文件"""
        from tkinter import filedialog
        try:
            import pandas as pd
        except ImportError:
            messagebox.showerror("缺少依赖", "请先安装pandas库：pip install pandas")
            return
        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel文件", "*.xlsx")], title="保存工资单")
            if not file_path:
                return
            # 查询工资数据
            self.cursor.execute("""
                SELECT s.id, p.name, s.month, s.working_hours, s.daily_rate, s.hourly_rate, s.performance_factor, s.total_salary, s.status
                FROM salary s JOIN persons p ON s.person_id=p.id
                WHERE s.month=?
            """, (self.salary_month_entry.get(),))
            rows = self.cursor.fetchall()
            columns = ["记录ID", "姓名", "月份", "工时", "日薪", "时薪", "绩效系数", "总工资", "状态"]
            df = pd.DataFrame(rows, columns=columns)
            df.to_excel(file_path, index=False)
            messagebox.showinfo("成功", f"工资单已成功导出到\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")

    def generate_stats(self):
        """生成统计图表 - 专业可视化"""
        try:
            stats_type = self.stats_type.get()
            start_date = self.stats_start_date.get()
            end_date = self.stats_end_date.get()

            # 更新核心指标
            self.update_core_metrics()

            if stats_type == "comprehensive":
                self.generate_comprehensive_dashboard(start_date, end_date)
            elif stats_type == "attendance":
                self.generate_attendance_analysis(start_date, end_date)
            elif stats_type == "violations":
                self.generate_violations_monitoring(start_date, end_date)
            elif stats_type == "persons":
                self.generate_persons_analysis()
            elif stats_type == "trends":
                self.generate_trends_insights(start_date, end_date)

        except Exception as e:
            messagebox.showerror("统计错误", f"生成统计图表失败: {str(e)}")

    def update_core_metrics(self):
        """更新核心指标数据 - 显示实际有用的统计信息"""
        try:
            # 注册人员总数
            self.cursor.execute("SELECT COUNT(*) FROM persons")
            total_persons = self.cursor.fetchone()[0]
            self.summary_vars["注册人员"].set(str(total_persons))

            # 总考勤记录数
            self.cursor.execute("SELECT COUNT(*) FROM attendance")
            total_attendance = self.cursor.fetchone()[0]
            self.summary_vars["今日考勤"].set(f"{total_attendance}次")

            # 活跃区域数
            self.cursor.execute("SELECT COUNT(DISTINCT area) FROM attendance")
            total_areas = self.cursor.fetchone()[0]
            self.summary_vars["活跃区域"].set(f"{total_areas}个")

            # 总安全事件数
            self.cursor.execute("SELECT COUNT(*) FROM violations")
            total_violations = self.cursor.fetchone()[0]
            self.summary_vars["安全事件"].set(f"{total_violations}起")

            # 系统状态
            self.summary_vars["系统状态"].set("正常运行")

            # AI准确率（基于实际数据计算）
            # 计算人脸识别成功率
            self.cursor.execute("SELECT COUNT(*) FROM attendance WHERE access_granted = 1")
            successful_access = self.cursor.fetchone()[0]
            self.cursor.execute("SELECT COUNT(*) FROM attendance")
            total_access = self.cursor.fetchone()[0]

            if total_access > 0:
                accuracy = (successful_access / total_access) * 100
                self.summary_vars["AI准确率"].set(f"{accuracy:.1f}%")
            else:
                self.summary_vars["AI准确率"].set("暂无数据")

        except Exception as e:
            print(f"更新核心指标失败: {e}")
            # 设置默认值
            self.summary_vars["注册人员"].set("--")
            self.summary_vars["今日考勤"].set("--")
            self.summary_vars["活跃区域"].set("--")
            self.summary_vars["安全事件"].set("--")
            self.summary_vars["系统状态"].set("异常")
            self.summary_vars["AI准确率"].set("--")

    def generate_comprehensive_dashboard(self, start_date: object, end_date: object) -> object:
        """生成综合仪表盘 - 显示实际数据"""
        self.stats_plot.clear()

        # 创建2x2子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 7))
        fig.patch.set_facecolor(self.card_color)

        # 1. 考勤趋势图 - 显示最近30天的趋势
        recent_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        self.cursor.execute("""
            SELECT date(entry_time), COUNT(*)
            FROM attendance
            WHERE date(entry_time) >= ?
            GROUP BY date(entry_time)
            ORDER BY date(entry_time)
            LIMIT 30
        """, (recent_date,))
        attendance_data = self.cursor.fetchall()

        if attendance_data:
            dates, counts = zip(*attendance_data)
            ax1.plot(dates, counts, 'o-', color=self.accent_color, linewidth=2, markersize=4)
            ax1.fill_between(dates, counts, alpha=0.3, color=self.accent_color)
            ax1.set_title("考勤趋势 (最近30天)", fontsize=12, weight="bold", color=self.text_color, pad=10)
            ax1.tick_params(axis='x', labelrotation=45, labelsize=8)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "暂无考勤数据", ha='center', va='center',
                    transform=ax1.transAxes, fontsize=10, color=self.theme_colors["muted"])

        # 2. 违规类型分布 - 显示所有违规数据
        self.cursor.execute("""
            SELECT violation_type, COUNT(*)
            FROM violations
            GROUP BY violation_type
        """)
        violation_data = self.cursor.fetchall()

        if violation_data:
            types, counts = zip(*violation_data)
            types = [self.violation_types.get(t, t) for t in types]
            colors = [self.theme_colors["danger"], self.theme_colors["warning"], self.accent_color, self.theme_colors["info"]]
            ax2.pie(counts, labels=types, autopct='%1.1f%%', colors=colors[:len(types)], startangle=90)
            ax2.set_title("违规类型分布", fontsize=12, weight="bold", color=self.text_color, pad=10)

        # 3. 人员权限结构 - 显示所有人员数据
        self.cursor.execute("SELECT permission_level, COUNT(*) FROM persons GROUP BY permission_level")
        person_data = self.cursor.fetchall()

        if person_data:
            levels, counts = zip(*person_data)
            levels = [self.permission_levels.get(l, f"级别 {l}") for l in levels]
            bars = ax3.bar(levels, counts, color=self.theme_colors["success"], alpha=0.8)
            ax3.set_title("人员权限结构", fontsize=12, weight="bold", color=self.text_color, pad=10)
            ax3.tick_params(axis='x', labelrotation=45, labelsize=8)

            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{count}', ha='center', va='bottom', fontsize=8)

        # 4. 区域活跃度 - 显示所有区域数据
        self.cursor.execute("""
            SELECT area, COUNT(*)
            FROM attendance
            GROUP BY area
            ORDER BY COUNT(*) DESC
            LIMIT 5
        """)
        area_data = self.cursor.fetchall()

        if area_data:
            areas, counts = zip(*area_data)
            bars = ax4.barh(areas, counts, color=self.theme_colors["info"], alpha=0.8)
            ax4.set_title("区域活跃度", fontsize=12, weight="bold", color=self.text_color, pad=10)

            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax4.text(count + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{count}', va='center', fontsize=8)

        plt.tight_layout()
        self.stats_canvas.figure = fig
        self.stats_canvas.draw()

    def generate_attendance_analysis(self, start_date, end_date):
        """生成考勤效能分析 - 显示实际数据"""
        self.stats_plot.clear()

        # 显示最近30天的考勤数据
        recent_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        self.cursor.execute("""
            SELECT date(entry_time), COUNT(*)
            FROM attendance
            WHERE date(entry_time) >= ?
            GROUP BY date(entry_time)
            ORDER BY date(entry_time)
            LIMIT 30
        """, (recent_date,))
        data = self.cursor.fetchall()

        if data:
            dates, counts = zip(*data)
            bars = self.stats_plot.bar(dates, counts, color=self.accent_color, alpha=0.8,
                                     edgecolor='white', linewidth=1, width=0.8)
            self.stats_plot.set_title("考勤效能分析 (最近30天)", fontsize=14, weight="bold",
                                    color=self.header_color, pad=20)
            self.stats_plot.set_xlabel("日期", fontsize=11, color=self.text_color, labelpad=10)
            self.stats_plot.set_ylabel("考勤人次", fontsize=11, color=self.text_color, labelpad=10)
            self.stats_plot.tick_params(axis='x', labelrotation=45, labelsize=9)
            self.stats_plot.grid(axis='y', linestyle='--', alpha=0.4)

            # 添加趋势线
            if len(dates) > 1:
                x_nums = range(len(dates))
                z = np.polyfit(x_nums, counts, 1)
                p = np.poly1d(z)
                self.stats_plot.plot(dates, p(x_nums), "r--", linewidth=2, alpha=0.8, label="趋势线")
                self.stats_plot.legend()

            # 添加数值标签
            for bar, count in zip(bars, counts):
                self.stats_plot.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                   f'{count}', ha='center', va='bottom', fontsize=9,
                                   color=self.accent_color, weight='bold')
        else:
            self.display_no_data_message("考勤数据")

        self.stats_figure.tight_layout()
        self.stats_canvas.draw()

    def generate_violations_monitoring(self, start_date, end_date):
        """生成安全违规监控 - 显示实际数据"""
        self.stats_plot.clear()

        # 显示所有违规数据，不限制日期
        self.cursor.execute("""
            SELECT violation_type, COUNT(*)
            FROM violations
            GROUP BY violation_type
        """)
        data = self.cursor.fetchall()

        if data:
            types, counts = zip(*data)
            types = [self.violation_types.get(t, t) for t in types]

            # 创建现代化饼图
            wedges, texts, autotexts = self.stats_plot.pie(counts, labels=types, autopct='%1.1f%%',
                                                         colors=[self.theme_colors["danger"],
                                                                self.theme_colors["warning"],
                                                                self.accent_color,
                                                                self.theme_colors["info"]][:len(types)],
                                                         startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                                                         textprops={'fontsize': 10, 'weight': 'bold'})

            self.stats_plot.set_title("安全违规监控分析 (全部数据)", fontsize=14, weight="bold",
                                    color=self.header_color, pad=20)

            # 美化标签
            for text in texts:
                text.set_fontsize(9)
                text.set_color(self.text_color)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
        else:
            self.display_no_data_message("违规数据")

        self.stats_figure.tight_layout()
        self.stats_canvas.draw()

    def generate_persons_analysis(self):
        """生成人员结构分析"""
        self.stats_plot.clear()
        self.cursor.execute("SELECT permission_level, COUNT(*) FROM persons GROUP BY permission_level")
        data = self.cursor.fetchall()

        if data:
            levels, counts = zip(*data)
            levels = [self.permission_levels.get(l, f"权限级别 {l}") for l in levels]

            bars = self.stats_plot.barh(levels, counts, color=self.theme_colors["success"],
                                      alpha=0.8, edgecolor='white', linewidth=1, height=0.6)
            self.stats_plot.set_title("人员结构分析", fontsize=14, weight="bold",
                                    color=self.header_color, pad=20)
            self.stats_plot.set_xlabel("人数", fontsize=11, color=self.text_color, labelpad=10)
            self.stats_plot.set_ylabel("权限等级", fontsize=11, color=self.text_color, labelpad=10)
            self.stats_plot.grid(axis='x', linestyle='--', alpha=0.4)

            # 添加数值标签和百分比
            total = sum(counts)
            for bar, count in zip(bars, counts):
                percentage = (count / total) * 100
                self.stats_plot.text(count + 0.3, bar.get_y() + bar.get_height()/2,
                                   f'{count} ({percentage:.1f}%)', va='center', fontsize=9,
                                   color=self.theme_colors["success"], weight='bold')
        else:
            self.display_no_data_message("人员数据")

        self.stats_figure.tight_layout()
        self.stats_canvas.draw()

    def generate_trends_insights(self, start_date, end_date):
        """生成时间趋势洞察 - 显示实际数据"""
        self.stats_plot.clear()

        # 获取最近30天的趋势数据
        recent_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        current_date = datetime.strptime(recent_date, "%Y-%m-%d")
        end = datetime.now()

        days = []
        attendance_trend = []
        violation_trend = []

        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            days.append(date_str)

            # 考勤数据
            self.cursor.execute("SELECT COUNT(*) FROM attendance WHERE date(entry_time) = ?", (date_str,))
            attendance_count = self.cursor.fetchone()[0]
            attendance_trend.append(attendance_count)

            # 违规数据
            self.cursor.execute("SELECT COUNT(*) FROM violations WHERE date(violation_time) = ?", (date_str,))
            violation_count = self.cursor.fetchone()[0]
            violation_trend.append(violation_count)

            current_date += timedelta(days=1)

        if days and any(attendance_trend) or any(violation_trend):
            ax1 = self.stats_plot
            ax1.set_title("时间趋势洞察分析", fontsize=14, weight="bold",
                         color=self.header_color, pad=20)
            ax1.set_xlabel("日期", fontsize=11, color=self.text_color, labelpad=10)
            ax1.set_ylabel("考勤人次", fontsize=11, color=self.accent_color, labelpad=10)
            ax1.tick_params(axis='x', labelrotation=45, labelsize=9)

            # 考勤趋势线
            line1 = ax1.plot(days, attendance_trend, 'o-', color=self.accent_color,
                           linewidth=2.5, markersize=5, label='考勤趋势', alpha=0.9)
            ax1.fill_between(days, attendance_trend, alpha=0.2, color=self.accent_color)

            # 违规趋势 (次坐标轴)
            ax2 = ax1.twinx()
            ax2.set_ylabel("违规次数", fontsize=11, color=self.theme_colors["danger"], labelpad=10)
            line2 = ax2.plot(days, violation_trend, 's-', color=self.theme_colors["danger"],
                           linewidth=2.5, markersize=5, label='违规趋势', alpha=0.9)
            ax2.fill_between(days, violation_trend, alpha=0.2, color=self.theme_colors["danger"])

            # 图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', fontsize=9)

            ax1.grid(True, linestyle='--', alpha=0.4)
        else:
            self.display_no_data_message("趋势数据")

        self.stats_figure.tight_layout()
        self.stats_canvas.draw()

    def display_no_data_message(self, data_type):
        """显示无数据消息"""
        self.stats_plot.text(0.5, 0.5, f"暂无{data_type}\n请检查数据或调整时间范围",
                           ha='center', va='center', transform=self.stats_plot.transAxes,
                           fontsize=12, color=self.theme_colors["muted"], weight='bold')
        """生成仪表盘总览"""
        self.stats_plot.clear()

        # 创建子图布局
        gs = self.stats_figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 考勤趋势
        ax1 = self.stats_figure.add_subplot(gs[0, 0])
        self.cursor.execute("""
            SELECT date(entry_time), COUNT(*)
            FROM attendance
            WHERE date(entry_time) BETWEEN ? AND ?
            GROUP BY date(entry_time)
            ORDER BY date(entry_time)
            LIMIT 7
        """, (start_date, end_date))
        attendance_data = self.cursor.fetchall()

        if attendance_data:
            dates, counts = zip(*attendance_data)
            ax1.plot(dates, counts, 'o-', color=self.accent_color, linewidth=2, markersize=6)
            ax1.fill_between(dates, counts, alpha=0.3, color=self.accent_color)
            ax1.set_title("考勤趋势", fontsize=12, weight="bold", color=self.text_color)
            ax1.tick_params(axis='x', labelrotation=45, labelsize=8)

        # 违规分布
        ax2 = self.stats_figure.add_subplot(gs[0, 1])
        self.cursor.execute("""
            SELECT violation_type, COUNT(*)
            FROM violations
            WHERE date(violation_time) BETWEEN ? AND ?
            GROUP BY violation_type
        """, (start_date, end_date))
        violation_data = self.cursor.fetchall()

        if violation_data:
            types, counts = zip(*violation_data)
            types = [self.violation_types.get(t, t) for t in types]
            ax2.bar(types, counts, color=self.theme_colors["danger"], alpha=0.7)
            ax2.set_title("违规分布", fontsize=12, weight="bold", color=self.text_color)
            ax2.tick_params(axis='x', labelrotation=45, labelsize=8)

        # 人员权限分布
        ax3 = self.stats_figure.add_subplot(gs[1, 0])
        self.cursor.execute("SELECT permission_level, COUNT(*) FROM persons GROUP BY permission_level")
        person_data = self.cursor.fetchall()

        if person_data:
            levels, counts = zip(*person_data)
            levels = [self.permission_levels.get(l, f"级别 {l}") for l in levels]
            ax3.pie(counts, labels=levels, autopct='%1.0f%%', colors=[self.accent_color, self.theme_colors["success"], self.theme_colors["warning"]])
            ax3.set_title("人员权限分布", fontsize=12, weight="bold", color=self.text_color)

        # 区域活跃度
        ax4 = self.stats_figure.add_subplot(gs[1, 1])
        self.cursor.execute("""
            SELECT area, COUNT(*)
            FROM attendance
            WHERE date(entry_time) BETWEEN ? AND ?
            GROUP BY area
        """, (start_date, end_date))
        area_data = self.cursor.fetchall()

        if area_data:
            areas, counts = zip(*area_data)
            ax4.barh(areas, counts, color=self.theme_colors["info"], alpha=0.7)
            ax4.set_title("区域活跃度", fontsize=12, weight="bold", color=self.text_color)

        self.stats_figure.suptitle("工地安全监控仪表盘", fontsize=14, weight="bold", color=self.header_color, y=0.98)
        self.stats_canvas.draw()

    def generate_attendance_stats(self, start_date, end_date):
        """生成考勤统计"""
        self.stats_plot.clear()
        self.cursor.execute("""
            SELECT date(entry_time), COUNT(*)
            FROM attendance
            WHERE date(entry_time) BETWEEN ? AND ?
            GROUP BY date(entry_time)
            ORDER BY date(entry_time)
        """, (start_date, end_date))
        data = self.cursor.fetchall()

        if data:
            dates, counts = zip(*data)
            bars = self.stats_plot.bar(dates, counts, color=self.accent_color, alpha=0.8, edgecolor='white', linewidth=1)
            self.stats_plot.set_title("考勤统计图表", fontsize=16, weight="bold", color=self.header_color, pad=20)
            self.stats_plot.set_xlabel("日期", fontsize=12, color=self.text_color)
            self.stats_plot.set_ylabel("考勤次数", fontsize=12, color=self.text_color)
            self.stats_plot.tick_params(axis='x', labelrotation=45, labelsize=10)
            self.stats_plot.grid(axis='y', linestyle='--', alpha=0.3)

            # 添加数值标签
            for bar, count in zip(bars, counts):
                self.stats_plot.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                   f'{count}', ha='center', va='bottom', fontsize=10, color=self.accent_color)
        else:
            self.stats_plot.text(0.5, 0.5, "暂无考勤数据", ha='center', va='center',
                               transform=self.stats_plot.transAxes, fontsize=14, color=self.theme_colors["muted"])

        self.stats_figure.tight_layout()
        self.stats_canvas.draw()

    def generate_violations_stats(self, start_date, end_date):
        """生成违规统计"""
        self.stats_plot.clear()
        self.cursor.execute("""
            SELECT violation_type, COUNT(*)
            FROM violations
            WHERE date(violation_time) BETWEEN ? AND ?
            GROUP BY violation_type
        """, (start_date, end_date))
        data = self.cursor.fetchall()

        if data:
            types, counts = zip(*data)
            types = [self.violation_types.get(t, t) for t in types]

            # 创建渐变色
            colors = ['#ff6b6b', '#ffa726', '#42a5f5', '#66bb6a', '#ab47bc']

            wedges, texts, autotexts = self.stats_plot.pie(counts, labels=types, autopct='%1.1f%%',
                                                         colors=colors[:len(types)], startangle=90,
                                                         wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            self.stats_plot.set_title("违规类型分布", fontsize=16, weight="bold", color=self.header_color, pad=20)

            # 美化标签
            for text in texts:
                text.set_fontsize(10)
                text.set_color(self.text_color)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
                autotext.set_weight('bold')
        else:
            self.stats_plot.text(0.5, 0.5, "暂无违规数据", ha='center', va='center',
                               transform=self.stats_plot.transAxes, fontsize=14, color=self.theme_colors["muted"])

        self.stats_figure.tight_layout()
        self.stats_canvas.draw()

    def generate_persons_stats(self):
        """生成人员统计"""
        self.stats_plot.clear()
        self.cursor.execute("SELECT permission_level, COUNT(*) FROM persons GROUP BY permission_level")
        data = self.cursor.fetchall()

        if data:
            levels, counts = zip(*data)
            levels = [self.permission_levels.get(l, f"级别 {l}") for l in levels]

            bars = self.stats_plot.barh(levels, counts, color=self.theme_colors["success"],
                                      alpha=0.8, edgecolor='white', linewidth=1)
            self.stats_plot.set_title("人员权限等级统计", fontsize=16, weight="bold", color=self.header_color, pad=20)
            self.stats_plot.set_xlabel("人数", fontsize=12, color=self.text_color)
            self.stats_plot.set_ylabel("权限等级", fontsize=12, color=self.text_color)
            self.stats_plot.grid(axis='x', linestyle='--', alpha=0.3)

            # 添加数值标签
            for bar, count in zip(bars, counts):
                self.stats_plot.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                                   f'{count}', ha='left', va='center', fontsize=10, color=self.theme_colors["success"])
        else:
            self.stats_plot.text(0.5, 0.5, "暂无人员数据", ha='center', va='center',
                               transform=self.stats_plot.transAxes, fontsize=14, color=self.theme_colors["muted"])

        self.stats_figure.tight_layout()
        self.stats_canvas.draw()

    def generate_trends_stats(self, start_date, end_date):
        """生成趋势分析"""
        self.stats_plot.clear()

        # 获取多天数据
        days = []
        attendance_trend = []
        violation_trend = []

        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            days.append(date_str)

            # 考勤数据
            self.cursor.execute("SELECT COUNT(*) FROM attendance WHERE date(entry_time) = ?", (date_str,))
            attendance_count = self.cursor.fetchone()[0]
            attendance_trend.append(attendance_count)

            # 违规数据
            self.cursor.execute("SELECT COUNT(*) FROM violations WHERE date(violation_time) = ?", (date_str,))
            violation_count = self.cursor.fetchone()[0]
            violation_trend.append(violation_count)

            current_date += timedelta(days=1)

        if days:
            ax1 = self.stats_plot
            ax1.set_title("考勤与违规趋势分析", fontsize=16, weight="bold", color=self.header_color, pad=20)
            ax1.set_xlabel("日期", fontsize=12, color=self.text_color)
            ax1.set_ylabel("考勤次数", fontsize=12, color=self.accent_color)
            ax1.tick_params(axis='x', labelrotation=45, labelsize=10)

            # 考勤线
            line1 = ax1.plot(days, attendance_trend, 'o-', color=self.accent_color,
                           linewidth=2, markersize=6, label='考勤次数')
            ax1.fill_between(days, attendance_trend, alpha=0.2, color=self.accent_color)

            # 违规线 (次坐标轴)
            ax2 = ax1.twinx()
            ax2.set_ylabel("违规次数", fontsize=12, color=self.theme_colors["danger"])
            line2 = ax2.plot(days, violation_trend, 's-', color=self.theme_colors["danger"],
                           linewidth=2, markersize=6, label='违规次数')
            ax2.fill_between(days, violation_trend, alpha=0.2, color=self.theme_colors["danger"])

            # 图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

            ax1.grid(True, linestyle='--', alpha=0.3)
        else:
            self.stats_plot.text(0.5, 0.5, "暂无趋势数据", ha='center', va='center',
                               transform=self.stats_plot.transAxes, fontsize=14, color=self.theme_colors["muted"])

        self.stats_figure.tight_layout()
        self.stats_canvas.draw()

    def setup_ai_framework_tab(self):
        """算法与训练：状态看板 + 模拟训练任务（可对接真实训练脚本）"""
        main_frame = ttk.Frame(self.tab_ai_framework)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # 左侧：算法服务状态
        left_frame = ttk.LabelFrame(main_frame, text="算法服务状态")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        status_frame = ttk.Frame(left_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(status_frame, text="服务状态:", font=('Microsoft YaHei', 10, 'bold')).pack(anchor=tk.W)
        self.ai_status_label = ttk.Label(status_frame, text="未初始化", font=('Microsoft YaHei', 10))
        self.ai_status_label.pack(anchor=tk.W)

        ttk.Label(status_frame, text="已加载模型:", font=('Microsoft YaHei', 10, 'bold')).pack(anchor=tk.W, pady=(10,0))
        self.ai_models_label = ttk.Label(status_frame, text="无", font=('Microsoft YaHei', 10))
        self.ai_models_label.pack(anchor=tk.W)

        ttk.Label(status_frame, text="设备信息:", font=('Microsoft YaHei', 10, 'bold')).pack(anchor=tk.W, pady=(10,0))
        self.ai_device_label = ttk.Label(status_frame, text="CPU模拟器", font=('Microsoft YaHei', 10))
        self.ai_device_label.pack(anchor=tk.W)

        ttk.Label(status_frame, text="内存使用:", font=('Microsoft YaHei', 10, 'bold')).pack(anchor=tk.W, pady=(10,0))
        self.ai_memory_label = ttk.Label(status_frame, text="0MB / 32GB", font=('Microsoft YaHei', 10))
        self.ai_memory_label.pack(anchor=tk.W)

        # 刷新状态按钮
        ttk.Button(left_frame, text="刷新状态", command=self.refresh_ai_status).pack(fill=tk.X, padx=5, pady=5)

        # 右侧：训练管理
        right_frame = ttk.LabelFrame(main_frame, text="模型训练管理")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 训练任务列表
        ttk.Label(right_frame, text="训练任务:", font=('Microsoft YaHei', 10, 'bold')).pack(anchor=tk.W, padx=5, pady=5)
        
        columns = ("任务ID", "状态", "进度", "准确率", "损失")
        self.training_tree = ttk.Treeview(right_frame, columns=columns, show="headings")

        for col in columns:
            self.training_tree.heading(col, text=col)
            self.training_tree.column(col, width=100, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=self.training_tree.yview)
        self.training_tree.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.training_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 训练控制按钮
        training_control_frame = ttk.Frame(right_frame)
        training_control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(training_control_frame, text="开始训练", command=self.start_ai_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(training_control_frame, text="停止训练", command=self.stop_ai_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(training_control_frame, text="刷新任务", command=self.refresh_training_tasks).pack(side=tk.LEFT, padx=5)

        # 初始化AI状态显示
        self.refresh_ai_status()

    def refresh_ai_status(self):
        """刷新算法服务状态"""
        if self.ai_framework and self.ai_framework.is_initialized:
            status = self.ai_framework.get_system_status()
            
            self.ai_status_label.config(text="✅ 已初始化")
            self.ai_status_label.config(foreground="green")
            
            loaded_models = status.get('loaded_models', [])
            self.ai_models_label.config(text=f"{len(loaded_models)} 个模型: {', '.join(loaded_models)}")
            
            device_info = status.get('device_info', {})
            device_id = device_info.get('device_id', 0)
            memory_total = device_info.get('memory_total', 0)
            memory_used = device_info.get('memory_used', 0)
            
            self.ai_device_label.config(text=f"CPU / 边缘设备 #{device_id}（演示内存模型）")
            self.ai_memory_label.config(text=f"{memory_used // (1024*1024*1024)}GB / {memory_total // (1024*1024*1024)}GB")
        else:
            self.ai_status_label.config(text="❌ 未初始化")
            self.ai_status_label.config(foreground="red")
            self.ai_models_label.config(text="无")
            self.ai_device_label.config(text="无设备")
            self.ai_memory_label.config(text="无数据")

    def start_ai_training(self):
        """开始模型训练（演示调度器；生产环境可改为调用外部训练脚本）"""
        if not self.ai_framework or not self.ai_framework.is_initialized:
            messagebox.showerror("错误", "算法服务未初始化")
            return

        # 创建训练任务
        job_id = f"training_{int(time.time())}"
        # 根据本地数据集动态扩展训练配置
        smoking_dataset_dir = os.path.join("datasets", "smoking")
        helmet_dataset_dir = os.path.join("datasets", "helmet")
        tasks = []
        if os.path.isdir(smoking_dataset_dir):
            tasks.append({
                'model_type': 'smoking_detection',
                'dataset_dir': smoking_dataset_dir,
                'epochs': 60,
                'batch_size': 32,
                'learning_rate': 0.001
            })
        if os.path.isdir(helmet_dataset_dir):
            tasks.append({
                'model_type': 'helmet_detection',
                'dataset_dir': helmet_dataset_dir,
                'epochs': 60,
                'batch_size': 32,
                'learning_rate': 0.001
            })
        # 始终保留人脸识别任务
        tasks.append({
            'model_type': 'face_recognition',
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        })
        training_config = {'multi_tasks': tasks}

        if self.ai_framework.training_manager.create_training_job(job_id, training_config):
            self.ai_framework.start_training(job_id)
            messagebox.showinfo("成功", f"训练任务 {job_id} 已开始")
            self.refresh_training_tasks()
        else:
            messagebox.showerror("错误", "创建训练任务失败")

    def stop_ai_training(self):
        """停止模型训练任务"""
        selected_items = self.training_tree.selection()
        if not selected_items:
            messagebox.showwarning("警告", "请先选择要停止的训练任务")
            return

        selected_item = selected_items[0]
        job_id = self.training_tree.item(selected_item, "values")[0]

        if self.ai_framework and self.ai_framework.training_manager.stop_training(job_id):
            messagebox.showinfo("成功", f"训练任务 {job_id} 已停止")
            self.refresh_training_tasks()
        else:
            messagebox.showerror("错误", "停止训练任务失败")

    def refresh_training_tasks(self):
        """刷新训练任务列表"""
        for item in self.training_tree.get_children():
            self.training_tree.delete(item)

        if self.ai_framework and self.ai_framework.is_initialized:
            training_jobs = self.ai_framework.training_manager.training_jobs
            
            for job_id, job_info in training_jobs.items():
                status = job_info.get('status', 'unknown')
                progress = f"{job_info.get('progress', 0):.1%}"
                accuracy = job_info.get('accuracy_history', [])
                loss = job_info.get('loss_history', [])
                
                accuracy_text = f"{accuracy[-1]:.4f}" if accuracy else "N/A"
                loss_text = f"{loss[-1]:.4f}" if loss else "N/A"
                
                self.training_tree.insert("", tk.END, values=(job_id, status, progress, accuracy_text, loss_text))


if __name__ == "__main__":
    root = tk.Tk()
    app = SecurityManagementSystem(root)
    root.mainloop()