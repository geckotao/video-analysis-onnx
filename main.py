# -*- coding: utf-8 -*-
# main.py
import sys
import os
import cv2
import time
import threading
import queue
import numpy as np
import configparser
import traceback
import datetime
import unicodedata
import re
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QCheckBox, QProgressBar, QTextEdit, QFileDialog, QMessageBox,
    QScrollArea, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QDialog,
    QSizePolicy, QSplashScreen, QListWidget, QListWidgetItem
)
from PySide6.QtGui import QPixmap, QImage, QIcon, QPen, QColor, QPainter, QFont
from PySide6.QtCore import Qt, QTimer, QPoint, Signal, QRect, QObject

# --- 依赖检查 ---
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    #设置 ONNX Runtime 日志级别，减少 EP Error 刷屏
    ort.set_default_logger_severity(3)
except ImportError:
    ONNX_AVAILABLE = False

#----获取基础路径的辅助函数----
def get_base_dir() -> Path:
    """兼容 Nuitka 打包后的路径"""
    if getattr(sys, 'frozen', False):
        # Nuitka 打包后
        return Path(sys.executable).parent
    else:
        return Path(__file__).parent

BASE_DIR = get_base_dir()

# --- 常量与配置 ---
CONFIG_PATH = str(BASE_DIR / 'set.ini')
LOG_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = LOG_DIR / "app_runtime.log"
ICON_PATH = BASE_DIR / "favicon.ico"
VIEWER_EXE_NAME = "viewer.exe"
NMS_IOU_THRESHOLD_DEFAULT = 0.35
INFERENCE_SIZE_DEFAULT = 640
CONF_THRESH_DEFAULT = 0.05
FILENAME_SANITIZE_REGEX = re.compile(r'\s+')
FILENAME_INVALID_CHARS = r'< > : " / \\ | ? *'
COCO_CLASS_MAPPING = {
    'person': '人', 'bicycle': '自行车', 'car': '小车', 'motorcycle': '摩托车',
    'airplane': '飞机', 'bus': '客车', 'train': '火车', 'truck': '货车', 'boat': '船',
    'traffic light': '交通灯', 'fire hydrant': '消防栓', 'stop sign': '停车标志',
    'parking meter': '停车计时器', 'bench': '长椅', 'bird': '鸟', 'cat': '猫', 'dog': '狗',
    'horse': '马', 'sheep': '羊', 'cow': '牛', 'elephant': '大象', 'bear': '熊',
    'zebra': '斑马', 'giraffe': '长颈鹿', 'backpack': '背包', 'umbrella': '雨伞',
    'handbag': '手提包', 'tie': '领带', 'suitcase': '行李箱', 'frisbee': '飞盘',
    'skis': '滑雪板', 'snowboard': '滑雪板', 'sports ball': '球类', 'kite': '风筝',
    'baseball bat': '棒球棒', 'baseball glove': '棒球手套', 'skateboard': '滑板',
    'surfboard': '冲浪板', 'tennis racket': '网球拍', 'bottle': '瓶子',
    'wine glass': '酒杯', 'cup': '杯子', 'fork': '叉子', 'knife': '刀', 'spoon': '勺子',
    'bowl': '碗', 'banana': '香蕉', 'apple': '苹果', 'sandwich': '三明治', 'orange': '橙子',
    'broccoli': '西兰花', 'carrot': '胡萝卜', 'hot dog': '热狗', 'pizza': '披萨',
    'donut': '甜甜圈', 'cake': '蛋糕', 'chair': '椅子', 'couch': '沙发', 'potted plant': '盆栽',
    'bed': '床', 'dining table': '餐桌', 'toilet': '马桶', 'tv': '电视', 'laptop': '笔记本电脑',
    'mouse': '鼠标', 'remote': '遥控器', 'keyboard': '键盘', 'cell phone': '手机',
    'microwave': '微波炉', 'oven': '烤箱', 'toaster': '烤面包机', 'sink': '水槽',
    'refrigerator': '冰箱', 'book': '书', 'clock': '时钟', 'vase': '花瓶', 'scissors': '剪刀',
    'teddy bear': '泰迪熊', 'hair drier': '吹风机', 'toothbrush': '牙刷'
}
COCO_CLASSES = list(COCO_CLASS_MAPPING.keys())
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 255),
    (255, 128, 0), (255, 0, 128), (0, 255, 128), (128, 255, 0), (128, 0, 255), (0, 128, 255)
]
COLORS_COUNT = len(COLORS)

# --- 辅助函数 ---

def sanitize_filename(filename: str) -> str:
    cleaned = "".join(
        c for c in filename
        if unicodedata.category(c) not in ("Cc", "Cf") and c not in FILENAME_INVALID_CHARS
    )
    cleaned = FILENAME_SANITIZE_REGEX.sub('_', cleaned.strip())
    return cleaned[:100] or "unnamed"

def cv2_to_qimage(cv_img: np.ndarray) -> QImage:
    if cv_img is None or cv_img.size == 0:
        return QImage()
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

def preprocess_image(img: np.ndarray, input_size: int = 640) -> Tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    img_padded = np.full((input_size, input_size, 3), 128, dtype=np.float32)
    dw = (input_size - new_w) // 2
    dh = (input_size - new_h) // 2
    img_padded[dh:dh+new_h, dw:dw+new_w, :] = img_resized
    img_padded = np.transpose(img_padded / 255.0, (2, 0, 1))
    img_padded = np.expand_dims(img_padded, axis=0).astype(np.float32)
    return img_padded, scale, dw, dh

def postprocess_yolo26(output: np.ndarray, img_shape: Tuple[int, int], input_size: int,
                       scale: float, dw: float, dh: float, conf_thres: float = 0.05,
                       iou_thres: float = 0.35) -> List[Dict[str, Any]]:
    predictions = np.squeeze(output[0])
    if len(predictions.shape) == 1:
        predictions = np.expand_dims(predictions, axis=0)
    scores = predictions[:, 4]
    valid_mask = scores > conf_thres
    predictions = predictions[valid_mask]
    scores = scores[valid_mask]
    if len(predictions) == 0:
        return []
    class_ids = predictions[:, 5].astype(int)
    boxes = predictions[:, :4]
    x1 = (boxes[:, 0] - dw) / scale
    y1 = (boxes[:, 1] - dh) / scale
    x2 = (boxes[:, 2] - dw) / scale
    y2 = (boxes[:, 3] - dh) / scale
    x1 = np.clip(x1, 0, img_shape[1])
    y1 = np.clip(y1, 0, img_shape[0])
    x2 = np.clip(x2, 0, img_shape[1])
    y2 = np.clip(y2, 0, img_shape[0])
    indices = cv2.dnn.NMSBoxes(
        np.array([x1, y1, x2 - x1, y2 - y1]).T.tolist(),
        scores.tolist(),
        conf_thres,
        iou_thres
    )
    detections = []
    if isinstance(indices, (np.ndarray, list)) and len(indices) > 0:
        indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
        for i in indices:
            i = int(i)
            class_id = int(class_ids[i])
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else 'unknown'
            detections.append({
                'x1': int(x1[i]),
                'y1': int(y1[i]),
                'x2': int(x2[i]),
                'y2': int(y2[i]),
                'score': float(scores[i]),
                'class_id': class_id,
                'class_name': class_name,
                'class_cn': COCO_CLASS_MAPPING.get(class_name, class_name)
            })
    return detections

# --- GPU 设备选择对话框 ---
class GPUDeviceDialog(QDialog):
    def __init__(self, parent, gpu_devices: List[Tuple[int, str]]):
        super().__init__(parent)
        self.setWindowTitle("选择推理设备")
        self.resize(550, 450)
        self.setModal(True)
        layout = QVBoxLayout(self)
        title_label = QLabel("请选择用于 ONNX 推理的设备：")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        self.device_list = QListWidget()
        self.device_list.setStyleSheet("""
            QListWidget {
                font-size: 13px;
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #0078D4;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #E5F3FF;
            }
        """)
        for device_id, device_name in gpu_devices:
            item_text = f"GPU {device_id}: {device_name}"
            item = QListWidgetItem(item_text)
            self.device_list.addItem(item)
        self.device_list.setCurrentRow(0)
        layout.addWidget(self.device_list)
        tip_label = QLabel("提示：选择推理耗时更短的GPU（通常独显更快）")
        tip_label.setStyleSheet("color: #666; font-size: 12px; margin-top: 5px;")
        tip_label.setWordWrap(True)
        layout.addWidget(tip_label)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.ok_btn = QPushButton("确定")
        self.ok_btn.setMinimumWidth(80)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("取消 (使用 CPU)")
        self.cancel_btn.setMinimumWidth(100)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def get_selected_device_id(self) -> int:
        return self.device_list.currentRow()

# --- 自定义控件 ---
class ClickableLabel(QLabel):
    clicked = Signal(QPoint)
    right_clicked = Signal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.position().toPoint())
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit()
        super().mousePressEvent(event)

class ROIDialog(QDialog):
    def __init__(self, parent, frame):
        super().__init__(parent)
        self.setWindowTitle("选择关注区域 (ROI)")
        self.resize(800, 600)
        self.frame = frame.copy()
        self.h, self.w = self.frame.shape[:2]
        self.points_orig = []
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pixmap = None
        self.image_label = ClickableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.clicked.connect(self._on_image_click)
        self.image_label.right_clicked.connect(self._on_image_right_click)
        self.image_label.setMinimumSize(0, 0)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(400, 300)
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        btn_layout = QHBoxLayout()
        self.confirm_btn = QPushButton("确认")
        self.reset_btn = QPushButton("重置")
        self.cancel_btn = QPushButton("取消")
        self.confirm_btn.clicked.connect(self.accept)
        self.reset_btn.clicked.connect(self.reset)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.confirm_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.update_display()

    def update_display(self):
        label_rect = self.image_label.contentsRect()
        label_w, label_h = label_rect.width(), label_rect.height()
        if label_w <= 1 or label_h <= 1:
            return
        scale_w = label_w / self.w
        scale_h = label_h / self.h
        self.scale = min(scale_w, scale_h)
        new_w = int(self.w * self.scale)
        new_h = int(self.h * self.scale)
        resized = cv2.resize(self.frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        qimg = cv2_to_qimage(resized)
        self.pixmap = QPixmap.fromImage(qimg)
        self.offset_x = (label_w - new_w) // 2
        self.offset_y = (label_h - new_h) // 2
        canvas = QPixmap(label_w, label_h)
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.drawPixmap(self.offset_x, self.offset_y, self.pixmap)
        painter.end()
        self.image_label.setPixmap(canvas)
        self.redraw_roi()

    def redraw_roi(self):
        if not hasattr(self, 'pixmap') or self.pixmap.isNull():
            return
        label_rect = self.image_label.contentsRect()
        canvas = QPixmap(label_rect.width(), label_rect.height())
        canvas.fill(Qt.black)
        painter = QPainter(canvas)
        painter.drawPixmap(self.offset_x, self.offset_y, self.pixmap)
        if len(self.points_orig) >= 1:
            points_on_label = [
                QPoint(int(x * self.scale + self.offset_x), int(y * self.scale + self.offset_y))
                for x, y in self.points_orig
            ]
            pen = QPen(QColor("lime"), 2)
            painter.setPen(pen)
            for i in range(len(points_on_label) - 1):
                painter.drawLine(points_on_label[i], points_on_label[i + 1])
            if len(points_on_label) >= 3:
                pen.setStyle(Qt.DashLine)
                painter.setPen(pen)
                painter.drawLine(points_on_label[-1], points_on_label[0])
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("red"))
            for pt in points_on_label:
                painter.drawEllipse(pt, 4, 4)
        painter.end()
        self.image_label.setPixmap(canvas)

    def _on_image_click(self, pos: QPoint):
        x_label = pos.x()
        y_label = pos.y()
        x_img = (x_label - self.offset_x) / self.scale
        y_img = (y_label - self.offset_y) / self.scale
        if 0 <= x_img < self.w and 0 <= y_img < self.h:
            self.points_orig.append((x_img, y_img))
            self.redraw_roi()

    def _on_image_right_click(self):
        if self.points_orig:
            self.points_orig.pop()
            self.redraw_roi()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(50, self.update_display)

    def reset(self):
        self.points_orig.clear()
        self.redraw_roi()

    def get_roi(self):
        if len(self.points_orig) < 3:
            return None
        return np.array([(int(round(x)), int(round(y))) for x, y in self.points_orig], dtype=np.int32)

# --- 主窗口 ---
class MainWindow(QMainWindow):
    # 所有 UI 更新都通过信号
    show_gpu_dialog_signal = Signal()
    buttons_signal = Signal(bool, bool, bool)
    processing_finished_signal = Signal(str)
    model_loaded_signal = Signal(bool)
    log_signal = Signal(str)
    status_signal = Signal(str)
    progress_signal = Signal(int)
    fps_signal = Signal(float)
    targets_signal = Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频目标检测截图工具（onnx模型版）V3.0 by geckotao")
        self.resize(1200, 800)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.class_names = {i: name for i, name in enumerate(COCO_CLASSES)}
        self.CHINESE_TO_ENGLISH = {v: k for k, v in COCO_CLASS_MAPPING.items()}
        self.class_checkboxes = []
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
            pass
        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_input_shape = None
        self._model_loaded = False
        self._preview_active = False
        self.paused = False
        self.stopped = False
        self.file_paths = []
        self.save_folder = ""
        self.roi = None
        self._processing_finished_normally = False
        self._generated_screenshot_paths = []
        self._stats_lock = threading.Lock()
        self._paths_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._stats = {'fps': 0.0, 'total_targets': 0, 'targets_by_class': {}}
        self._progress_current = 0
        self._progress_total = 1
        self._processing_file_index = -1
        self._total_files = 1
        self._model_load_triggered = False
        self._preview_queue = queue.Queue(maxsize=10)
        self.save_queue = queue.Queue(maxsize=100)
        self._gpu_device_id = 0
        self._gpu_device_chosen = False
        self._gpu_devices = []
        self._test_input_name = 'images'
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        self.setup_ui()
        self.load_configuration()
        
        # 连接所有信号到槽函数
        self.show_gpu_dialog_signal.connect(self._show_gpu_device_dialog)
        self.log_signal.connect(self._update_log_ui)
        self.status_signal.connect(self._update_status_ui)
        self.progress_signal.connect(self._update_progress_ui)
        self.fps_signal.connect(self._update_fps_ui)
        self.targets_signal.connect(self._update_targets_ui)
        self.buttons_signal.connect(self._update_buttons_ui)
        
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats_periodically)
        self.stats_timer.start(500)
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self._update_preview_display)
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self._update_progress_bar)
        self.progress_timer.start(100)
        self.model_loaded_signal.connect(self.on_model_loaded)
        self.processing_finished_signal.connect(self.on_processing_finished)

    # ========== 线程安全的 UI 槽函数 ==========
    def _update_log_ui(self, message: str):
        if hasattr(self, 'log_text') and self.log_text:
            self.log_text.append(message)
            self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _update_status_ui(self, text: str):
        if hasattr(self, 'status_label') and self.status_label:
            self.status_label.setText(text)

    def _update_progress_ui(self, value: int):
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.setValue(value)

    def _update_fps_ui(self, fps: float):
        if hasattr(self, 'fps_label') and self.fps_label:
            self.fps_label.setText(f"FPS(每秒处理帧数): {fps:.1f}")

    def _update_targets_ui(self, targets: int):
        if hasattr(self, 'targets_label') and self.targets_label:
            self.targets_label.setText(f"截图数（未去重）: {targets}")

    def _update_buttons_ui(self, start_en: bool, pause_en: bool, stop_en: bool):
        """【关键】通过信号更新按钮状态"""
        if hasattr(self, 'start_button'):
            self.start_button.setEnabled(start_en)
        if hasattr(self, 'pause_button'):
            self.pause_button.setEnabled(pause_en)
        if hasattr(self, 'stop_button'):
            self.stop_button.setEnabled(stop_en)

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, '_model_load_triggered', False):
            self._model_load_triggered = True
            QTimer.singleShot(50, self.start_model_loading)

    def start_model_loading(self):
        if not ONNX_AVAILABLE:
            self.update_status("环境错误：未安装 onnxruntime")
            self.buttons_signal.emit(False, False, False)
            return
        self.update_status("模型加载中...")
        self.buttons_signal.emit(False, False, False)
        self.log_message("开始加载 ONNX 模型...")
        threading.Thread(target=self._load_in_thread, daemon=True).start()

    def _get_directml_devices(self) -> List[Tuple[int, str]]:
        devices = []
        if sys.platform != 'win32':
            return devices
        test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        for device_id in range(2):  # 只测试前 2 个设备,一般电脑只有一个集显及独显
            test_session = None
            try:
                test_session = ort.InferenceSession(
                    self.MODEL_PATH,
                    providers=['DmlExecutionProvider'],
                    provider_options=[{'device_id': device_id}]
                )
                actual_providers = test_session.get_providers()
                if 'DmlExecutionProvider' in actual_providers:
                    input_name = test_session.get_inputs()[0].name
                    self._test_input_name = input_name
                    inference_times = []
                    for _ in range(3):
                        start_infer = time.time()
                        test_session.run(None, {input_name: test_input})
                        inference_times.append((time.time() - start_infer) * 1000)
                    avg_infer_time = sum(inference_times) / len(inference_times)
                    device_name = f"GPU {device_id} (测试推理时间：{avg_infer_time:.0f}ms)"
                    devices.append((device_id, device_name))
                    self.log_message(f"GPU ID {device_id}: {avg_infer_time:.0f}ms")
            except Exception:
                pass  # 静默失败
            finally:
                if test_session is not None:
                    del test_session
        if not devices:
            devices = [(0, "默认 DirectML GPU 设备")]
        return devices

    def _load_in_thread(self):
        try:
            success = self.load_model()
            self.model_loaded_signal.emit(success)
        except Exception as e:
            self.log_message(f"[模型加载异常] {e}")
            self.model_loaded_signal.emit(False)

    def _show_gpu_device_dialog(self):
        try:
            
            self.log_message("DirectML 可用的 GPU 设备：")
            for device_id, gpu_name in self._gpu_devices:
                self.log_message(f"   {gpu_name}")
            
            self.log_message("等待用户选择推理设备...")
            dialog = GPUDeviceDialog(self, self._gpu_devices)
            if dialog.exec():
                selected_id = dialog.get_selected_device_id()
                self._gpu_device_id = selected_id
                self.log_message(f"用户选择设备 ID {selected_id}")
            else:
                self._gpu_device_id = -1
                self.log_message("用户取消选择，将使用 CPU 推理")
        except Exception as e:
            self.log_message(f"[对话框错误] {e}")
            self._gpu_device_id = -1
        finally:
            self._gpu_device_chosen = True

    def on_model_loaded(self, success: bool):
        if success:
            self._model_loaded = True
            self.update_status("模型加载完成，可开始处理")
            self.buttons_signal.emit(True, False, False)  # 【修复】通过信号
            self.log_message("程序初始化完成")
        else:
            self.update_status("模型加载失败，请检查配置或重启程序")
            QMessageBox.critical(self, "模型加载失败",
                "YOLO26 ONNX 模型加载失败，请检查：\n"
                "1. 模型文件路径是否正确\n"
                "2. onnxruntime 是否安装\n"
                "3. DirectML 依赖是否完整")
            self.buttons_signal.emit(False, False, False)  # 【修复】通过信号

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignTop)
        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("选择视频文件:"))
        self.video_entry = QLineEdit()
        video_browse_btn = QPushButton("浏览")
        video_browse_btn.clicked.connect(self.select_video)
        video_layout.addWidget(self.video_entry)
        video_layout.addWidget(video_browse_btn)
        left_layout.addLayout(video_layout)
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("截图保存路径:"))
        self.save_path_entry = QLineEdit()
        save_browse_btn = QPushButton("浏览")
        save_browse_btn.clicked.connect(self.select_save_path)
        save_layout.addWidget(self.save_path_entry)
        save_layout.addWidget(save_browse_btn)
        left_layout.addLayout(save_layout)
        classes_group = QGroupBox("选择目标类别")
        classes_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.select_all_classes)
        deselect_btn = QPushButton("取消已选")
        deselect_btn.clicked.connect(self.deselect_all_classes)
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(deselect_btn)
        classes_layout.addLayout(btn_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        grid_layout = QGridLayout(scroll_content)
        self.class_checkboxes = []
        default_selected = {'bicycle', 'car', 'motorcycle', 'bus', 'truck'}
        for i, (idx, name) in enumerate(self.class_names.items()):
            display = COCO_CLASS_MAPPING.get(name, name)
            cb = QCheckBox(display)
            cb.setChecked(name in default_selected)
            self.class_checkboxes.append(cb)
            row, col = divmod(i, 5)
            grid_layout.addWidget(cb, row, col)
        scroll_area.setWidget(scroll_content)
        classes_layout.addWidget(scroll_area)
        classes_group.setLayout(classes_layout)
        left_layout.addWidget(classes_group)
        options_group = QGroupBox("处理选项")
        options_layout = QGridLayout()
        options_layout.addWidget(QLabel("跳帧数:"), 0, 0)
        self.speed_entry = QLineEdit("23")
        options_layout.addWidget(self.speed_entry, 0, 1)
        options_layout.addWidget(QLabel("(1=逐帧，值越大越快)"), 0, 2)
        options_layout.addWidget(QLabel("置信度阈值:"), 1, 0)
        self.confidence_entry = QLineEdit("0.1")
        options_layout.addWidget(self.confidence_entry, 1, 1)
        options_layout.addWidget(QLabel("(0.0-1.0)"), 1, 2)
        self.only_moving_var = QCheckBox("只对移动目标截图")
        self.only_moving_var.setChecked(True)
        self.annotate_var = QCheckBox("在截图中标注目标")
        options_layout.addWidget(self.only_moving_var, 2, 0)
        options_layout.addWidget(self.annotate_var, 2, 1)
        roi_layout = QHBoxLayout()
        self.roi_button = QPushButton("选取关注区域")
        self.roi_button.clicked.connect(self.select_roi)
        self.roi_label = QLabel("未选取关注区域")
        self.roi_label.setStyleSheet("color: gray;")
        roi_layout.addWidget(self.roi_button)
        roi_layout.addWidget(self.roi_label)
        options_layout.addLayout(roi_layout, 3, 0)
        preview_layout = QHBoxLayout()
        self.preview_button = QPushButton("开启实时预览")
        self.preview_button.clicked.connect(self.toggle_preview)
        self.preview_status = QLabel("预览：关闭")
        self.preview_status.setStyleSheet("color: gray;")
        preview_layout.addWidget(self.preview_button)
        preview_layout.addWidget(self.preview_status)
        options_layout.addLayout(preview_layout, 3, 1)
        options_group.setLayout(options_layout)
        left_layout.addWidget(options_group)
        ctrl_layout = QHBoxLayout()
        self.start_button = QPushButton("开始处理")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_processing)
        self.pause_button = QPushButton("暂停处理")
        self.pause_button.clicked.connect(self.pause_processing)
        self.pause_button.setEnabled(False)
        self.stop_button = QPushButton("停止处理")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        ctrl_layout.addWidget(self.start_button)
        ctrl_layout.addWidget(self.pause_button)
        ctrl_layout.addWidget(self.stop_button)
        left_layout.addLayout(ctrl_layout)
        stats_group = QGroupBox("实时统计")
        stats_layout = QVBoxLayout()
        self.fps_label = QLabel("FPS(每秒处理帧数): 0.0")
        self.targets_label = QLabel("截图总数（未去重）: 0")
        stats_layout.addWidget(self.fps_label)
        stats_layout.addWidget(self.targets_label)
        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(6)
        right_layout.setContentsMargins(8, 8, 8, 8)
        self.status_label = QLabel("模型加载中...")
        self.status_label.setStyleSheet("color: #666; font-weight: bold;")
        right_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        right_layout.addWidget(self.progress_bar)
        log_group = QGroupBox("处理日志")
        log_group.setFixedHeight(150)
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group)
        preview_group = QGroupBox("实时预览")
        preview_layout_widget = QVBoxLayout(preview_group)
        self.preview_label = QLabel("无实时预览画面")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(600, 400)
        self.preview_label.setStyleSheet("background-color: black; color: white;")
        preview_layout_widget.addWidget(self.preview_label)
        right_layout.addWidget(preview_group, stretch=1)
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 2)

    # ========== 公共方法（子线程可调用）==========
    def log_message(self, message: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        except Exception:
            pass
        self.log_signal.emit(formatted)

    def update_status(self, text: str):
        self.status_signal.emit(text)

    def _save_worker(self):
        while True:
            try:
                task = self.save_queue.get()
                if task is None:
                    break
                image, save_path = task
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                ret, buffer = cv2.imencode('.jpg', image)
                if ret:
                    with open(save_path, 'wb') as f:
                        f.write(buffer)
                self.save_queue.task_done()
            except Exception as e:
                self.log_message(f"[保存错误] {e}")

    def launch_result_viewer(self, task_file_path: str):
        try:
            if getattr(sys, 'frozen', False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).parent
            viewer_exe = base_dir / VIEWER_EXE_NAME
            if not viewer_exe.exists():
                self.log_message(f"错误：viewer.exe 不存在于 {viewer_exe}")
                QMessageBox.critical(self, "错误", f"未找到 viewer.exe，请确保它与主程序在同一目录！")
                return
            subprocess.Popen([str(viewer_exe), "--task", task_file_path])
            self.log_message(f"已启动 viewer.exe，加载任务文件：{task_file_path}")
        except Exception as e:
            self.log_message(f"启动 viewer.exe 失败：{e}")
            QMessageBox.critical(self, "错误", f"无法启动结果查看器：\n{str(e)}")

    def load_configuration(self):
        config = configparser.ConfigParser()
        if not os.path.exists(CONFIG_PATH):
            config['Settings'] = {
                'model_path': './models/yolo26x.onnx',
                'nms_iou_threshold': str(NMS_IOU_THRESHOLD_DEFAULT),
                'inference_size': str(INFERENCE_SIZE_DEFAULT),
                'conf_thres': str(CONF_THRESH_DEFAULT),
                'batch_size': '1',
                'roi_iou_threshold': '0.2',
                'movement_iou_threshold': '0.6',
                'movement_relative_threshold': '0.02',
                'movement_consecutive_frames': '5',
                'movement_stable_seconds': '5'
            }
            try:
                with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                    config.write(f)
                self.log_message(f"已创建默认配置文件 {CONFIG_PATH}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法创建配置文件：{e}")
                return False
        else:
            config.read(CONFIG_PATH, encoding='utf-8')
        self.MODEL_PATH = config.get('Settings', 'model_path', fallback='./models/yolo26x.onnx')
        try:
            self.NMS_IOU_THRESHOLD = float(config.get('Settings', 'nms_iou_threshold', fallback=str(NMS_IOU_THRESHOLD_DEFAULT)))
            self.NMS_IOU_THRESHOLD = max(0.0, min(1.0, self.NMS_IOU_THRESHOLD))
        except ValueError:
            self.NMS_IOU_THRESHOLD = NMS_IOU_THRESHOLD_DEFAULT
        try:
            self.INFERENCE_SIZE = int(config.get('Settings', 'inference_size', fallback=str(INFERENCE_SIZE_DEFAULT)))
            self.INFERENCE_SIZE = max(320, min(1280, self.INFERENCE_SIZE))
        except ValueError:
            self.INFERENCE_SIZE = INFERENCE_SIZE_DEFAULT
        try:
            self.CONF_THRESH = float(config.get('Settings', 'conf_thres', fallback=str(CONF_THRESH_DEFAULT)))
            self.CONF_THRESH = max(0.0, min(1.0, self.CONF_THRESH))
        except ValueError:
            self.CONF_THRESH = CONF_THRESH_DEFAULT
        self.BATCH_SIZE = 1
        try:
            self.ROI_IOU_THRESHOLD = float(config.get('Settings', 'roi_iou_threshold', fallback='0.2'))
        except ValueError:
            self.ROI_IOU_THRESHOLD = 0.2
        try:
            self.MOVEMENT_IOU_THRESHOLD = float(config.get('Settings', 'movement_iou_threshold', fallback='0.6'))
        except ValueError:
            self.MOVEMENT_IOU_THRESHOLD = 0.6
        try:
            self.MOVEMENT_RELATIVE_THRESHOLD = float(config.get('Settings', 'movement_relative_threshold', fallback='0.02'))
        except ValueError:
            self.MOVEMENT_RELATIVE_THRESHOLD = 0.02
        try:
            self.MOVEMENT_CONSECUTIVE_FRAMES = int(config.get('Settings', 'movement_consecutive_frames', fallback='5'))
        except ValueError:
            self.MOVEMENT_CONSECUTIVE_FRAMES = 5
        try:
            self.MOVEMENT_STABLE_SECONDS = float(config.get('Settings', 'movement_stable_seconds', fallback='5'))
        except ValueError:
            self.MOVEMENT_STABLE_SECONDS = 5
        self.log_message(f"配置文件读取成功：推理分辨率={self.INFERENCE_SIZE}, conf_thresh={self.CONF_THRESH}")
        return True

    def load_model(self):
        if not ONNX_AVAILABLE:
            self.log_message("onnxruntime 未安装")
            QMessageBox.critical(self, "错误", "缺少 onnxruntime！")
            return False
        try:
            if not os.path.exists(self.MODEL_PATH):
                raise FileNotFoundError(f"模型文件不存在：{self.MODEL_PATH}")
            self.log_message(f"ONNX Runtime 版本：{ort.__version__}")
            available_providers = ort.get_available_providers()
            self.log_message(f"发现执行提供器：{available_providers}，开始测试是否可用...")
            providers = []
            provider_options = []
            selected_device_id = 0
            selected_gpu_name = "未知"
            if 'DmlExecutionProvider' in available_providers:
                directml_devices = self._get_directml_devices()
                self._gpu_devices = directml_devices
                self._gpu_device_chosen = False
                self._gpu_device_id = 0
                self.show_gpu_dialog_signal.emit()
                while not self._gpu_device_chosen:
                    time.sleep(0.05)
                    QApplication.processEvents()
                if self._gpu_device_id >= 0:
                    try:
                        test_session = ort.InferenceSession(
                            self.MODEL_PATH,
                            providers=['DmlExecutionProvider'],
                            provider_options=[{'device_id': self._gpu_device_id}]
                        )
                        actual_providers = test_session.get_providers()
                        if 'DmlExecutionProvider' in actual_providers:
                            selected_device_id = self._gpu_device_id
                            selected_gpu_name = directml_devices[selected_device_id][1] if selected_device_id < len(directml_devices) else f"GPU {selected_device_id}"
                            providers = ['DmlExecutionProvider']
                            provider_options = [{'device_id': selected_device_id}]
                            self.log_message(f"GPU ID {selected_device_id} 验证成功")
                        else:
                            raise Exception(f"GPU ID {selected_device_id} 未使用 DirectML")
                        del test_session
                    except Exception as e:
                        self.log_message(f"GPU ID {self._gpu_device_id} 不可用：{str(e)[:100]}")
                        self.log_message("回退到 CPU 推理")
                        providers = ['CPUExecutionProvider']
                        provider_options = [{}]
                        selected_gpu_name = "CPU"
                else:
                    providers = ['CPUExecutionProvider']
                    provider_options = [{}]
                    selected_gpu_name = "CPU"
                    self.log_message("使用 CPU 推理")
                
                if providers and selected_gpu_name != "CPU":
                    self.log_message(f"程序将使用 GPU ID {selected_device_id} 推理")

            elif 'CPUExecutionProvider' in available_providers:
                providers = ['CPUExecutionProvider']
                provider_options = [{}]
                
                self.log_message("未检测到 DirectML，使用 CPU 推理")
                selected_gpu_name = "CPU"
            else:
                
                self.log_message("错误：没有可用的执行提供器！")
                return False
            self.onnx_session = ort.InferenceSession(
                self.MODEL_PATH,
                providers=providers,
                provider_options=provider_options if len(provider_options) == len(providers) else None
            )
            actual_providers = self.onnx_session.get_providers()

            if 'DmlExecutionProvider' in actual_providers:
                
                self.log_message("状态：GPU 加速已启用")
                self.log_message(f"使用GPU: ID {selected_device_id}")
            else:
                self.log_message("状态： 未使用 GPU 加速（CPU 推理）")
            input_info = self.onnx_session.get_inputs()[0]
            output_info = self.onnx_session.get_outputs()[0]
            self.onnx_input_name = input_info.name
            self.onnx_input_shape = input_info.shape
            self.log_message(f"模型【{self.MODEL_PATH}】加载成功")
            self.log_message(f"输入形状：{self.onnx_input_shape} | 输出形状：{output_info.shape}")
            return True
        except Exception as e:
            self.log_message(f"模型加载失败：{e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "错误", f"模型加载失败：\n{e}")
            return False

    def select_video(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择视频文件", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.mpeg *.mpg *.m1v *.m2v *.vob *.ts *.m2ts *.mts *.ps)")
        if paths:
            self.file_paths = paths
            display = ", ".join(paths[:2]) + f", ... (+{len(paths)-2})" if len(paths) > 3 else ", ".join(paths)
            self.video_entry.setText(display)
            self.log_message(f"已选择 {len(self.file_paths)} 个视频文件")

    def select_save_path(self):
        folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if folder:
            self.save_folder = folder
            self.save_path_entry.setText(folder)
            try:
                test = os.path.join(folder, "test.tmp")
                open(test, 'w').close()
                os.remove(test)
                self.log_message("路径验证：具有写入权限，支持中文路径")
            except Exception as e:
                QMessageBox.warning(self, "权限警告", f"保存路径可能有问题：{e}")

    def select_all_classes(self):
        for cb in self.class_checkboxes:
            cb.setChecked(True)
        self.log_message("全选所有类别")

    def deselect_all_classes(self):
        for cb in self.class_checkboxes:
            cb.setChecked(False)
        self.log_message("取消已选目标")

    def toggle_preview(self):
        if not self.file_paths:
            self.log_message("请先选择视频文件")
            return
        if self._preview_active:
            self._preview_active = False
            self.preview_button.setText("开启实时预览")
            self.preview_status.setText("预览：关闭")
            self.preview_status.setStyleSheet("color: gray;")
            self.preview_label.setText("预览已关闭")
            self.preview_timer.stop()
            self.log_message("实时预览已关闭")
        else:
            self._preview_active = True
            self.preview_button.setText("关闭实时预览")
            self.preview_status.setText("预览：开启")
            self.preview_status.setStyleSheet("color: orange;")
            self.log_message("实时预览已开启")
            self.preview_timer.start(30)

    def _update_preview_display(self):
        if not self._preview_active:
            return
        try:
            item = self._preview_queue.get_nowait()
            if isinstance(item, tuple) and len(item) == 3:
                frame, frame_idx, annotate_info = item
                if frame is not None:
                    if self.roi is not None:
                        cv2.polylines(frame, [self.roi], isClosed=True, color=(0, 255, 0), thickness=2)
                    for (x1, y1, x2, y2, class_display, score) in annotate_info:
                        color = COLORS[hash(class_display) % COLORS_COUNT]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        class_en = self.CHINESE_TO_ENGLISH.get(class_display, class_display)
                        text = f"{class_en} {score:.2f}"
                        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        text_x = x1
                        text_y = y1 - 10 if y1 - 10 > 0 else y1 + text_h + 10
                        cv2.rectangle(frame, (text_x, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), color, -1)
                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    qimg = cv2_to_qimage(frame)
                    pixmap = QPixmap.fromImage(qimg)
                    if not pixmap.isNull():
                        self.preview_label.setPixmap(pixmap.scaled(
                            self.preview_label.size(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        ))
                    else:
                        self.preview_label.setText("预览图像无效")
                else:
                    self.preview_label.setText("视频结束")
        except queue.Empty:
            pass
        except Exception as e:
            self.log_message(f"[预览错误] {e}")

    def select_roi(self):
        if not self.file_paths:
            self.log_message("错误：请先选择视频文件")
            return
        cap = None
        try:
            cap = cv2.VideoCapture(self.file_paths[0])
            ret, frame = cap.read()
            if not ret:
                self.log_message("错误：无法读取视频首帧")
                return
            dialog = ROIDialog(self, frame)
            if dialog.exec():
                roi = dialog.get_roi()
                if roi is not None and len(roi) >= 3:
                    area = cv2.contourArea(roi)
                    if area > 1.0:
                        self.roi = roi
                        self.roi_label.setText("关注区域已设置")
                        self.roi_label.setStyleSheet("color: orange;")
                        self.roi_button.setText("取消选取")
                        try:
                            self.roi_button.clicked.disconnect()
                        except:
                            pass
                        self.roi_button.clicked.connect(self.cancel_roi)
                        self.log_message(f"成功设置 ROI，面积：{area:.1f} 像素")
                    else:
                        QMessageBox.warning(self, "ROI 无效", "ROI 面积过小")
                else:
                    QMessageBox.warning(self, "ROI 无效", "至少需要 3 个点")
            else:
                self.log_message("ROI 选择已取消")
        except Exception as e:
            self.log_message(f"[ROI 错误] {e}")
            QMessageBox.critical(self, "错误", f"ROI 选择失败：{e}")
        finally:
            if cap is not None:
                cap.release()

    def cancel_roi(self):
        self.roi = None
        self.roi_label.setText("未选取关注区域")
        self.roi_label.setStyleSheet("color: gray;")
        self.roi_button.setText("选取关注区域")
        try:
            self.roi_button.clicked.disconnect()
        except:
            pass
        self.roi_button.clicked.connect(self.select_roi)
        self.log_message("已取消关注区域")

    def _box_iou_simple(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2
        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _is_point_in_roi(self, x, y):
        if self.roi is None:
            return True
        try:
            pt = (int(round(x)), int(round(y)))
            result = cv2.pointPolygonTest(self.roi, pt, measureDist=False)
            return result >= 0
        except Exception as e:
            self.log_message(f"[ROI 安全防护] 检查点 ({x:.1f},{y:.1f}) 时出错：{e}")
            return True

    def _is_target_moving_enhanced(self, object_key, current_box, current_time, prev_positions, stable_since, movement_buffer):
        prev_data = prev_positions.get(object_key)
        if prev_data is None:
            prev_positions[object_key] = (current_box, current_time)
            return True
        prev_box, prev_time = prev_data
        time_diff = current_time - prev_time
        if time_diff <= 0:
            return False
        if time_diff >= 0.3:
            cx1, cy1 = (prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2
            cx2, cy2 = (current_box[0] + current_box[2]) / 2, (current_box[1] + current_box[3]) / 2
            dx, dy = abs(cx2 - cx1), abs(cy2 - cy1)
            w = (current_box[2] - current_box[0] + prev_box[2] - prev_box[0]) / 2 + 1e-6
            h = (current_box[3] - current_box[1] + prev_box[3] - prev_box[1]) / 2 + 1e-6
            speed_x = dx / time_diff / w
            speed_y = dy / time_diff / h
            if speed_x > 1.0 or speed_y > 1.0:
                stable_since.pop(object_key, None)
                movement_buffer[object_key] = 0
                prev_positions[object_key] = (current_box, current_time)
                return True
        iou = self._box_iou_simple(prev_box, current_box)
        if iou > self.MOVEMENT_IOU_THRESHOLD:
            if object_key not in stable_since:
                stable_since[object_key] = current_time
            prev_positions[object_key] = (current_box, current_time)
            return False
        x1, y1, x2, y2 = current_box
        x1_p, y1_p, x2_p, y2_p = prev_box
        w_avg = ((x2 - x1) + (x2_p - x1_p)) / 2 + 1e-6
        h_avg = ((y2 - y1) + (y2_p - y1_p)) / 2 + 1e-6
        dx = abs((x1 + x2) / 2 - (x1_p + x2_p) / 2)
        dy = abs((y1 + y2) / 2 - (y1_p + y2_p) / 2)
        rel_dx, rel_dy = dx / w_avg, dy / h_avg
        moved = rel_dx > self.MOVEMENT_RELATIVE_THRESHOLD or rel_dy > self.MOVEMENT_RELATIVE_THRESHOLD
        if object_key not in movement_buffer:
            movement_buffer[object_key] = 0
        movement_buffer[object_key] = movement_buffer[object_key] + 1 if moved else 0
        last_stable = stable_since.get(object_key, -10)
        recently_stable = (current_time - last_stable) < self.MOVEMENT_STABLE_SECONDS
        if recently_stable and movement_buffer[object_key] >= self.MOVEMENT_CONSECUTIVE_FRAMES:
            stable_since.pop(object_key, None)
            result = True
        elif not recently_stable:
            result = movement_buffer[object_key] >= self.MOVEMENT_CONSECUTIVE_FRAMES
        else:
            result = False
        prev_positions[object_key] = (current_box, current_time)
        return result

    def start_processing(self):
        if not self.file_paths:
            self.log_message("错误：未选择视频文件")
            return
        if not self.save_folder:
            self.log_message("错误：未选择保存路径")
            return
        try:
            test = os.path.join(self.save_folder, "test.tmp")
            open(test, 'w').close()
            os.remove(test)
        except Exception as e:
            QMessageBox.critical(self, "路径错误", f"保存路径不可写：{e}")
            return
        selected = [self.class_names[i] for i, cb in enumerate(self.class_checkboxes) if cb.isChecked()]
        if not selected:
            self.log_message("错误：未选择任何目标类")
            return
        try:
            skip_frames = max(1, min(64, int(self.speed_entry.text())))
        except ValueError:
            skip_frames = 1
        try:
            conf = max(0.0, min(1.0, float(self.confidence_entry.text())))
        except ValueError:
            conf = self.CONF_THRESH
        only_moving = self.only_moving_var.isChecked()
        annotate = self.annotate_var.isChecked()
        self.paused = False
        self.stopped = False
        self._processing_finished_normally = False
        with self._paths_lock:
            self._generated_screenshot_paths.clear()
        # 【修复】通过信号更新按钮
        self.buttons_signal.emit(False, True, True)
        self.log_message("开始处理...")
        threading.Thread(target=self.process_videos, args=(
            self.file_paths, self.save_folder, selected, skip_frames, only_moving, annotate, conf
        ), daemon=True).start()

    def pause_processing(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.setText("继续处理")
            self.log_message("处理已暂停")
            self.update_status("处理已暂停...")
        else:
            self.pause_button.setText("暂停处理")
            self.log_message("处理已继续")
            self.update_status("处理中...")

    def stop_processing(self):
        if not self.stopped:
            self.stopped = True
            self.paused = False
            self.log_message("正在停止处理...")
            self.update_status("正在停止...")
            while not self.save_queue.empty():
                try:
                    self.save_queue.get_nowait()
                    self.save_queue.task_done()
                except:
                    break
            # 【修复】通过信号更新按钮
            self.buttons_signal.emit(True, False, False)

    def process_videos(self, file_paths, save_folder, selected_classes, skip_frames, only_moving, annotate_objects, confidence_threshold):
        try:
            total_files = len(file_paths)
            self._total_files = total_files
            for file_idx, file_path in enumerate(file_paths):
                if self.stopped:
                    break
                self._processing_file_index = file_idx
                self.current_frame = 0
                cap_temp = cv2.VideoCapture(file_path)
                total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_temp.release()
                self._progress_total = total_frames
                prev_positions = {}
                stable_since = {}
                movement_buffer = {}
                frame_queue = queue.Queue(maxsize=10)
                def read_frames():
                    cap = cv2.VideoCapture(file_path)
                    frame_idx = 0
                    try:
                        while cap.isOpened() and not self.stopped:
                            while self.paused:
                                time.sleep(0.01)
                            if self.stopped:
                                break
                            ret = cap.grab()
                            if not ret:
                                break
                            if frame_idx % skip_frames == 0:
                                ret, frame = cap.retrieve()
                                if ret:
                                    try:
                                        frame_queue.put((frame.copy(), frame_idx), timeout=1)
                                    except queue.Full:
                                        time.sleep(0.01)
                            frame_idx += 1
                    finally:
                        cap.release()
                    frame_queue.put(None)
                reader_thread = threading.Thread(target=read_frames, daemon=True)
                reader_thread.start()
                last_inference_time = time.time()
                frame_count_for_fps = 0
                while True:
                    if self.stopped:
                        break
                    try:
                        item = frame_queue.get(timeout=2)
                        if item is None:
                            break
                    except queue.Empty:
                        continue
                    orig_frame, frame_count = item
                    input_size = self.INFERENCE_SIZE
                    img_preprocessed, scale, dw, dh = preprocess_image(orig_frame, input_size)
                    start_time = time.time()
                    output = self.onnx_session.run(None, {self.onnx_input_name: img_preprocessed})
                    frame_count_for_fps += 1
                    current_time = time.time()
                    if current_time > last_inference_time and frame_count_for_fps > 0:
                        with self._stats_lock:
                            self._stats['fps'] = frame_count_for_fps / (current_time - last_inference_time)
                            self.fps_signal.emit(self._stats['fps'])
                        last_inference_time = current_time
                        frame_count_for_fps = 0
                    detections = postprocess_yolo26(
                        output, orig_frame.shape, input_size, scale, dw, dh,
                        conf_thres=confidence_threshold, iou_thres=self.NMS_IOU_THRESHOLD
                    )
                    filtered_dets = []
                    for det in detections:
                        class_name = det['class_name']
                        if class_name not in selected_classes:
                            continue
                        cx, cy = (det['x1'] + det['x2']) / 2, (det['y1'] + det['y2']) / 2
                        if self.roi is not None:
                            if not self._is_point_in_roi(cx, cy):
                                continue
                        filtered_dets.append(det)
                    seen_targets_in_frame = set()
                    preview_annotate_info = []
                    targets_to_save = []
                    current_time = time.time()
                    for det in filtered_dets:
                        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                        class_name = det['class_name']
                        class_display = det['class_cn']
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        target_key = (class_name, int(round(cx)), int(round(cy)))
                        if target_key in seen_targets_in_frame:
                            continue
                        seen_targets_in_frame.add(target_key)
                        object_key = f"{class_name}_{int(cx)}_{int(cy)}"
                        should_save = True
                        if only_moving:
                            is_moving = self._is_target_moving_enhanced(
                                object_key, [x1, y1, x2, y2], current_time,
                                prev_positions, stable_since, movement_buffer
                            )
                            should_save = is_moving
                        if should_save:
                            targets_to_save.append((x1, y1, x2, y2, class_display, det['score']))
                            preview_annotate_info.append((x1, y1, x2, y2, class_display, float(det['score'])))
                    if targets_to_save:
                        video_name = Path(file_path).stem
                        for i, (x1, y1, x2, y2, class_display, score) in enumerate(targets_to_save):
                            if annotate_objects:
                                annotated_frame = orig_frame.copy()
                                color = COLORS[hash(class_display) % COLORS_COUNT]
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                class_en = self.CHINESE_TO_ENGLISH.get(class_display, class_display)
                                text = f"{class_en} {score:.2f}"
                                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                text_x = x1
                                text_y = y1 - 10 if y1 - 10 > 0 else y1 + text_h + 10
                                cv2.rectangle(annotated_frame, (text_x, text_y - text_h - 2), (text_x + text_w + 2, text_y + 2), color, -1)
                                cv2.putText(annotated_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                image_to_save = annotated_frame
                            else:
                                image_to_save = orig_frame.copy()
                            safe_class = sanitize_filename(class_display)
                            filename = f"{video_name}_frame{frame_count}_{safe_class}.jpg"
                            save_path = os.path.join(save_folder, filename)
                            try:
                                self.save_queue.put_nowait((image_to_save, save_path))
                                with self._stats_lock:
                                    self._stats['total_targets'] += 1
                                    self._stats['targets_by_class'][class_display] = \
                                        self._stats['targets_by_class'].get(class_display, 0) + 1
                                    self.targets_signal.emit(self._stats['total_targets'])
                                with self._paths_lock:
                                    self._generated_screenshot_paths.append(os.path.abspath(save_path))
                            except queue.Full:
                                pass
                    if self._preview_active:
                        try:
                            if self._preview_queue.full():
                                self._preview_queue.get_nowait()
                            self._preview_queue.put_nowait((orig_frame.copy(), frame_count, preview_annotate_info))
                        except queue.Full:
                            pass
                    self._progress_current = frame_count
                reader_thread.join(timeout=2)
        except Exception as e:
            self.log_message(f"处理出错：{e}\n{traceback.format_exc()}")
            self.update_status(f"错误：{str(e)}")
        finally:
            # 【关键修复】通过信号更新按钮，不在子线程直接操作 UI
            self.buttons_signal.emit(True, False, False)
            if self.stopped:
                self.update_status("处理已停止")
                self.log_message("处理已停止")
            else:
                self._processing_finished_normally = True
                self.update_status("所有文件处理完成！")
                task_file = self._generate_task_summary_file()
                if task_file:
                    self.processing_finished_signal.emit(task_file)
                else:
                    self.processing_finished_signal.emit("")

    def _update_progress_bar(self):
        with self._progress_lock:
            if self._processing_finished_normally:
                self.progress_signal.emit(100)
                self.status_signal.emit("所有文件处理完成！")
                return
            if self.stopped:
                status_text = "处理已停止"
            elif self._total_files <= 0:
                status_text = "准备中..."
            else:
                file_progress = self._progress_current / self._progress_total if self._progress_total > 0 else 0
                overall_progress = (self._processing_file_index + file_progress) / self._total_files
                progress_percent = min(99, int(overall_progress * 100 + 0.5))
                self.progress_signal.emit(progress_percent)
                status_text = f"处理文件 {self._processing_file_index + 1}/{self._total_files} | 帧：{self._progress_current}/{self._progress_total}"
            if not self._processing_finished_normally:
                self.status_signal.emit(status_text)

    def update_stats_periodically(self):
        with self._stats_lock:
            fps = self._stats['fps']
            total_targets = self._stats['total_targets']
        self.fps_signal.emit(fps)
        self.targets_signal.emit(total_targets)

    def _generate_task_summary_file(self) -> str:
        with self._paths_lock:
            paths = list(self._generated_screenshot_paths)
        if not paths:
            self.log_message("无截图生成，跳过任务记录文件创建。")
            return ""
        unique_paths = list(dict.fromkeys(paths))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task_filename = f"任务_{timestamp}.txt"
        task_filepath = os.path.join(self.save_folder, task_filename)
        try:
            with open(task_filepath, "w", encoding="utf-8") as f:
                f.write("[Processed Videos]\n")
                for video_path in self.file_paths:
                    f.write(f"{os.path.abspath(video_path)}\n")
                f.write("\n[Generated Screenshots]\n")
                for img_path in unique_paths:
                    f.write(f"{img_path}\n")
            self.log_message(f"任务记录文件已生成（已去重）: {task_filepath}")
            return os.path.abspath(task_filepath)
        except Exception as e:
            self.log_message(f"生成任务记录文件失败：{e}")
            QMessageBox.warning(self, "警告", f"无法创建任务记录文件：\n{e}")
            return ""

    def closeEvent(self, event):
        self.stopped = True
        try:
            self.save_queue.put(None)
            self.save_thread.join(timeout=2)
        except:
            pass
        event.accept()

    def on_processing_finished(self, task_file_path: str):
        if hasattr(self, '_view_results_button') and self._view_results_button:
            self._view_results_button.deleteLater()
            self._view_results_button = None
        if not task_file_path:
            return
        self._view_results_button = QPushButton("查看结果")
        self._view_results_button.clicked.connect(
            lambda: self.launch_result_viewer(task_file_path)
        )
        right_layout = self.status_label.parentWidget().layout()
        if right_layout:
            right_layout.insertWidget(2, self._view_results_button)
        else:
            self.log_message("警告：无法定位右侧布局")

def main():
    app = QApplication(sys.argv)
    splash_pix = QPixmap(400, 200)
    splash_pix.fill(Qt.white)
    painter = QPainter(splash_pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor(66, 133, 244), 3))
    painter.drawRect(10, 10, 380, 180)
    painter.setPen(QColor(33, 33, 33))
    painter.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
    painter.drawText(QRect(0, 40, 400, 40), Qt.AlignCenter, "视频目标检测截图工具")
    painter.setFont(QFont("Microsoft YaHei", 10))
    painter.drawText(QRect(0, 90, 400, 30), Qt.AlignCenter, "V3.0 by geckotao")
    painter.setPen(QColor(220, 50, 47))
    painter.drawText(QRect(0, 130, 400, 30), Qt.AlignCenter, "⏳ 程序初始化中...")
    painter.end()
    splash = QSplashScreen(splash_pix)
    splash.show()
    app.processEvents()
    window = MainWindow()
    window.showMaximized()
    splash.finish(window)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()