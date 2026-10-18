import sys, os

# --- Fix for module import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
# --- End of fix ---

import numpy as np
import torch
from PIL import Image
from collections import OrderedDict, Counter
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QTextEdit, QMessageBox, QHBoxLayout, QProgressBar,
    QFrame, QScrollArea, QGridLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QGroupBox, QSizePolicy,
    QComboBox, QSpinBox
)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon, QPainter, QBrush, QPen
from PyQt5.QtCore import Qt, QRunnable, QThreadPool, pyqtSlot, QObject, pyqtSignal, QSize, QTimer

# === Import your training module ===
try:
    import Training_pytourch as Training
except ImportError:
    try:
        import Training
    except ImportError as e:
        print(f"FATAL: Could not import Training module: {e}")
        sys.exit(1)

# --- Import TTA function ---
try:
    _tta_variants = getattr(Training, "_tta_variants")
except AttributeError:
    print("Warning: TTA function not found. Using fallback.")
    _tta_variants = lambda x, tta: [x]


# ============================================================
# WORKER SIGNALS & THREADING
# ============================================================

class WorkerSignals(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(tuple)
    finished = pyqtSignal()
    progress = pyqtSignal(int, str)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as e:
            import traceback
            self.signals.error.emit((type(e), e, traceback.format_exc()))
        finally:
            self.signals.finished.emit()


# ============================================================
# CUSTOM WIDGETS
# ============================================================

class ProbabilityBar(QWidget):
    """Custom widget to display a probability bar with label"""
    def __init__(self, class_name, probability, color, parent=None):
        super().__init__(parent)
        self.class_name = class_name
        self.probability = probability
        self.color = color
        self.setMinimumHeight(40)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.fillRect(self.rect(), QColor("#F8F9FA"))
        bar_width = int(self.width() * 0.7 * self.probability)
        painter.fillRect(10, 10, bar_width, 20, QColor(self.color))
        painter.setPen(QPen(QColor("#DEE2E6"), 1))
        painter.drawRect(10, 10, int(self.width() * 0.7), 20)
        painter.setPen(QColor("#212529"))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(15, 25, self.class_name)
        painter.setFont(QFont("Arial", 9))
        painter.drawText(int(self.width() * 0.75), 25, f"{self.probability*100:.1f}%")


class ModernCard(QFrame):
    """Modern card container with shadow effect"""
    def __init__(self, title=None, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #E9ECEF;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        if title:
            title_label = QLabel(title)
            title_label.setFont(QFont("Arial", 12, QFont.Bold))
            title_label.setStyleSheet("color: #495057; border: none;")
            layout.addWidget(title_label)
            
            divider = QFrame()
            divider.setFrameShape(QFrame.HLine)
            divider.setStyleSheet("background-color: #DEE2E6; border: none;")
            divider.setFixedHeight(1)
            layout.addWidget(divider)
        
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)
        self.setLayout(layout)


class StatusIndicator(QLabel):
    """Animated status indicator"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self.status = "idle"
        self.update_style()
        
    def set_status(self, status):
        self.status = status
        self.update_style()
        
    def update_style(self):
        colors = {
            "idle": "#6C757D",
            "loading": "#FFC107",
            "success": "#28A745",
            "error": "#DC3545"
        }
        color = colors.get(self.status, "#6C757D")
        self.setStyleSheet(f"background-color: {color}; border-radius: 6px;")


# ============================================================
# MAIN APPLICATION
# ============================================================

class ModernDRCAD(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diabetic Retinopathy AI Diagnosis System")
        self.setMinimumSize(1400, 900)
        
        # Modern color scheme
        self.colors = {
            'primary': '#4A90E2',
            'success': '#28A745',
            'danger': '#DC3545',
            'warning': '#FFC107',
            'info': '#17A2B8',
            'light': '#F8F9FA',
            'dark': '#343A40',
            'background': '#F0F2F5'
        }
        
        self.setup_app_style()
        
        # === Paths for Raspberry Pi ===
        self.WORK_DIR = "/home/masou/Desktop/Work"
        self.CSV_LABELS_PATH = "/home/masou/Desktop/archive(1)/trainLabels_cropped_2Lines.csv"
        self.OUTLIER_FILE_PATH = "/home/masou/Desktop/EDA_Results_DeepLearning/outliers.csv"
        self.SUMMARY_CSV_PATH = os.path.join(self.WORK_DIR, "summary_metrics.csv")
        
        # RPi 5 Optimization: Start with best_only, user can change
        self.ENSEMBLE_STRATEGY = "best_only"
        self.MIN_QWK_THRESHOLD = 0.75
        self.MIN_ACCURACY_THRESHOLD = 0.80
        
        self.LABELS = [0, 1, 2, 3, 4]
        self.CLASS_NAMES = {
            0: "No DR", 
            1: "Mild NPDR", 
            2: "Moderate NPDR", 
            3: "Severe NPDR", 
            4: "Proliferative DR"
        }
        self.CLASS_COLORS = {
            0: "#28A745",
            1: "#FFC107",
            2: "#FD7E14",
            3: "#DC3545",
            4: "#6F42C1"
        }
        
        self.CLASS_DESCRIPTIONS = {
            0: "No signs of diabetic retinopathy detected. Regular monitoring recommended.",
            1: "Mild non-proliferative diabetic retinopathy. Early stage with microaneurysms.",
            2: "Moderate non-proliferative diabetic retinopathy. More widespread retinal damage.",
            3: "Severe non-proliferative diabetic retinopathy. Extensive retinal damage requiring close monitoring.",
            4: "Proliferative diabetic retinopathy. Advanced stage with new blood vessel growth."
        }
        
        self.IMG_SIZE = getattr(Training, "DEFAULT_IMG_SIZE", 224)
        
        # RPi 5 Optimization
        print("RPi 5 Optimization: Forcing CPU and setting 4 threads.")
        try:
            torch.set_num_threads(4)
        except Exception as e:
            print(f"Warning: Could not set torch threads: {e}")
        self.DEVICE = torch.device("cpu")
        
        _, self.val_transform = getattr(Training, "get_transforms")(self.IMG_SIZE)
        
        self.models = {}
        self.model_weights = {}
        self.model_metrics = {}
        self.ground_truth_labels = {}
        self.outlier_paths_set = set()
        self.current_image_path = None
        self.models_loaded = False
        self.prediction_history = []
        self.loaded_models_list = []  # Track which models are loaded
        
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(2)  # Limit threads on RPi
        print(f"Thread pool: {self.threadpool.maxThreadCount()} threads")
        
        self.init_ui()
        self.load_ground_truth_labels()
        self.load_outlier_list()
        self.start_model_loading()
        
    def setup_app_style(self):
        """Set application-wide styles"""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.colors['background']};
                color: {self.colors['dark']};
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }}
            QPushButton {{
                background-color: {self.colors['primary']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #357ABD;
            }}
            QPushButton:pressed {{
                background-color: #2D6BA3;
            }}
            QPushButton:disabled {{
                background-color: #ADB5BD;
                color: #6C757D;
            }}
            QProgressBar {{
                border: 2px solid #DEE2E6;
                border-radius: 8px;
                text-align: center;
                background-color: white;
                height: 25px;
            }}
            QProgressBar::chunk {{
                background-color: {self.colors['primary']};
                border-radius: 6px;
            }}
            QTextEdit {{
                background-color: white;
                border: 1px solid #DEE2E6;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }}
            QComboBox {{
                background-color: white;
                border: 1px solid #DEE2E6;
                border-radius: 6px;
                padding: 6px;
                font-size: 12px;
            }}
            QSpinBox {{
                background-color: white;
                border: 1px solid #DEE2E6;
                border-radius: 6px;
                padding: 6px;
            }}
        """)
    
    def init_ui(self):
        """Initialize the main UI"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
        self.setLayout(main_layout)
        
    def create_header(self):
        """Create the application header"""
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self.colors['primary']}, stop:1 #5BA3F5);
                border-radius: 12px;
                padding: 10px;
            }}
        """)
        
        layout = QHBoxLayout()
        
        title = QLabel("🔬 Diabetic Retinopathy AI Diagnosis")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: white; background: transparent; border: none;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Status indicator
        status_container = QWidget()
        status_container.setStyleSheet("background: transparent; border: none;")
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        self.status_indicator = StatusIndicator()
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: white; background: transparent; font-size: 12px;")
        
        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_container)
        
        # Memory usage label
        self.memory_label = QLabel("RAM: --")
        self.memory_label.setStyleSheet("color: white; background: transparent; font-size: 12px;")
        layout.addWidget(self.memory_label)
        
        # Start memory monitoring
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.start(5000)  # Update every 5 seconds
        
        header.setLayout(layout)
        header.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return header
    
    def update_memory_usage(self):
        """Update memory usage display"""
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            self.memory_label.setText(f"RAM: {mem_mb:.0f} MB")
        except:
            pass
        
    def create_left_panel(self):
        """Create left panel with image and controls"""
        container = ModernCard()
        
        # Settings card
        settings_card = ModernCard("⚙️ Model Settings")
        
        # Strategy selector
        strategy_layout = QHBoxLayout()
        strategy_label = QLabel("Strategy:")
        strategy_label.setStyleSheet("border: none; font-weight: bold;")
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["best_only", "top_performers", "adaptive", "all"])
        self.strategy_combo.setCurrentText(self.ENSEMBLE_STRATEGY)
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        strategy_layout.addWidget(strategy_label)
        strategy_layout.addWidget(self.strategy_combo)
        settings_card.content_layout.addLayout(strategy_layout)
        
        # TTA selector
        tta_layout = QHBoxLayout()
        tta_label = QLabel("TTA Steps:")
        tta_label.setStyleSheet("border: none; font-weight: bold;")
        self.tta_spin = QSpinBox()
        self.tta_spin.setRange(1, 8)
        self.tta_spin.setValue(1)
        tta_layout.addWidget(tta_label)
        tta_layout.addWidget(self.tta_spin)
        settings_card.content_layout.addLayout(tta_layout)
        
        # Reload models button
        self.btn_reload = QPushButton("🔄 Reload Models")
        self.btn_reload.clicked.connect(self.reload_models)
        self.btn_reload.setEnabled(False)
        settings_card.content_layout.addWidget(self.btn_reload)
        
        container.content_layout.addWidget(settings_card)
        
        # Image display
        image_container = QFrame()
        image_container.setStyleSheet("""
            QFrame {
                background-color: #212529;
                border-radius: 8px;
                border: 2px solid #DEE2E6;
            }
        """)
        image_layout = QVBoxLayout(image_container)
        
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("color: #6C757D; font-size: 16px; background: transparent; border: none;")
        image_layout.addWidget(self.image_label)
        
        container.content_layout.addWidget(image_container)
        
        # Image info
        self.image_info_label = QLabel("")
        self.image_info_label.setStyleSheet("color: #6C757D; font-size: 11px; border: none;")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        container.content_layout.addWidget(self.image_info_label)
        
        # Control buttons
        button_layout = QGridLayout()
        button_layout.setSpacing(10)
        
        self.btn_load = QPushButton("📁 Upload Image")
        self.btn_load.clicked.connect(self.load_image)
        self.btn_load.setEnabled(False)
        
        self.btn_camera = QPushButton("📷 Camera Capture")
        self.btn_camera.clicked.connect(self.capture_camera)
        self.btn_camera.setEnabled(False)
        
        self.btn_test = QPushButton("🔍 Analyze Image")
        self.btn_test.clicked.connect(self.run_test)
        self.btn_test.setEnabled(False)
        self.btn_test.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['success']};
                font-size: 15px;
                padding: 15px;
            }}
            QPushButton:hover {{
                background-color: #218838;
            }}
            QPushButton:disabled {{
                background-color: #ADB5BD;
            }}
        """)
        
        button_layout.addWidget(self.btn_load, 0, 0)
        button_layout.addWidget(self.btn_camera, 0, 1)
        button_layout.addWidget(self.btn_test, 1, 0, 1, 2)
        
        container.content_layout.addLayout(button_layout)
        
        # Model status card
        model_card = ModernCard("Models Status")
        self.model_status_text = QTextEdit()
        self.model_status_text.setReadOnly(True)
        self.model_status_text.setMaximumHeight(150)
        self.model_status_text.setText("⏳ Loading models...")
        model_card.content_layout.addWidget(self.model_status_text)
        container.content_layout.addWidget(model_card)
        
        container.content_layout.addStretch()
        
        return container
        
    def create_right_panel(self):
        """Create right panel with results tabs"""
        tabs = QTabWidget()
        
        # Tab 1: Prediction Results
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        # Diagnosis card
        self.diagnosis_card = ModernCard("Diagnosis Result")
        self.diagnosis_label = QLabel("Awaiting analysis...")
        self.diagnosis_label.setAlignment(Qt.AlignCenter)
        self.diagnosis_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #495057; border: none; padding: 30px;")
        self.diagnosis_card.content_layout.addWidget(self.diagnosis_label)
        
        self.confidence_label = QLabel("")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("font-size: 18px; color: #6C757D; border: none;")
        self.diagnosis_card.content_layout.addWidget(self.confidence_label)
        
        results_layout.addWidget(self.diagnosis_card)
        
        # Probability distribution
        prob_card = ModernCard("Class Probabilities")
        self.prob_container = QWidget()
        self.prob_layout = QVBoxLayout(self.prob_container)
        self.prob_layout.setSpacing(5)
        prob_card.content_layout.addWidget(self.prob_container)
        results_layout.addWidget(prob_card)
        
        # Model votes
        votes_card = ModernCard("Individual Model Predictions")
        self.votes_table = QTableWidget()
        self.votes_table.setColumnCount(4)
        self.votes_table.setHorizontalHeaderLabels(["Model", "Prediction", "Confidence", "Used"])
        self.votes_table.horizontalHeader().setStretchLastSection(True)
        self.votes_table.setEditTriggers(QTableWidget.NoEditTriggers)
        votes_card.content_layout.addWidget(self.votes_table)
        results_layout.addWidget(votes_card)
        
        tabs.addTab(results_tab, "📊 Results")
        
        # Tab 2: Detailed Log
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        tabs.addTab(log_tab, "📝 Detailed Log")
        
        # Tab 3: History
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["Time", "Image", "Prediction", "Confidence", "Ground Truth"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        history_layout.addWidget(self.history_table)
        
        clear_history_btn = QPushButton("🗑️ Clear History")
        clear_history_btn.clicked.connect(self.clear_history)
        clear_history_btn.setMaximumWidth(200)
        history_layout.addWidget(clear_history_btn)
        
        tabs.addTab(history_tab, "📜 History")
        
        return tabs

    # ============================================================
    # MODEL LOADING
    # ============================================================
    
    def load_ground_truth_labels(self):
        """Load ground truth labels from CSV"""
        try:
            import pandas as pd
        except ImportError:
            self.log("FATAL: Pandas not installed. Ground truth loading failed.")
            return

        if os.path.exists(self.CSV_LABELS_PATH):
            try:
                df = pd.read_csv(self.CSV_LABELS_PATH)
                if 'image' in df.columns and 'label' in df.columns:
                    df['id_code'] = df['image'].apply(lambda x: os.path.splitext(x)[0])
                    self.ground_truth_labels = pd.Series(df.label.values, index=df.id_code).to_dict()
                    self.log(f"✅ Loaded {len(self.ground_truth_labels)} ground truth labels")
            except Exception as e:
                self.log(f"⚠️ Could not load ground truth CSV: {e}")
    
    def load_outlier_list(self):
        """Loads the list of outlier image paths from the EDA csv."""
        if os.path.exists(self.OUTLIER_FILE_PATH):
            try:
                import pandas as pd
                df = pd.read_csv(self.OUTLIER_FILE_PATH)
                if 'path' in df.columns:
                    self.outlier_paths_set = set(os.path.normpath(p) for p in df['path'].tolist())
                    self.log(f"✅ Loaded {len(self.outlier_paths_set)} outliers for pre-check.")
                else:
                    self.log("⚠️ Outlier CSV found but missing 'path' column.")
            except Exception as e:
                self.log(f"⚠️ Warning: Could not load outlier CSV. {e}")
        else:
            self.log(f"ℹ️ Outlier file not found at {self.OUTLIER_FILE_PATH}. Skipping outlier pre-check.")
    
    def on_strategy_changed(self, strategy):
        """Handle strategy change"""
        self.ENSEMBLE_STRATEGY = strategy
        self.log(f"Strategy changed to: {strategy}")
        self.btn_reload.setEnabled(True)
    
    def reload_models(self):
        """Reload models with new strategy"""
        reply = QMessageBox.question(
            self, "Reload Models",
            f"Reload models with '{self.ENSEMBLE_STRATEGY}' strategy?\n\nThis will clear current models from memory.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear existing models
            self.models.clear()
            self.model_weights.clear()
            self.loaded_models_list.clear()
            self.models_loaded = False
            import gc
            gc.collect()
            
            # Reload
            self.start_model_loading()
    
    def start_model_loading(self):
        """Start background model loading"""
        self.status_indicator.set_status("loading")
        self.status_label.setText("Loading models...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Loading models...")
        
        worker = Worker(self.load_models_based_on_strategy)
        worker.signals.result.connect(self.on_models_loaded)
        worker.signals.error.connect(self.on_models_load_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        worker.signals.progress.connect(self.on_loading_progress)
        
        self.threadpool.start(worker)
    
    def load_models_based_on_strategy(self):
        """Load models based on strategy"""
        candidate_models = self.list_candidate_models()
        models_to_load = []
        
        if os.path.exists(self.SUMMARY_CSV_PATH):
            try:
                import pandas as pd
                df = pd.read_csv(self.SUMMARY_CSV_PATH)
                df = df[df['model'] != 'Ensemble']
                
                for _, row in df.iterrows():
                    if row['model'] in candidate_models:
                        self.model_metrics[row['model']] = {
                            'qwk': row.get('qwk', 0),
                            'accuracy': row.get('accuracy', 0),
                            'f1': row.get('f1_weighted', 0)
                        }
                
                # Mock models dict for filtering
                self.models = {m: None for m in self.model_metrics.keys()}
                models_to_load = self.filter_models_by_strategy()
                self.models = {}
                
                if not models_to_load:
                    print("Warning: Filtering resulted in no models. Loading best model.")
                    if self.model_metrics:
                        best = max(self.model_metrics.items(), key=lambda x: x[1]['qwk'])[0]
                        models_to_load = [best]
                    else:
                        models_to_load = candidate_models[:1]
                    
            except Exception as e:
                print(f"Warning: Could not read summary CSV: {e}. Loading best model.")
                models_to_load = candidate_models[:1] if candidate_models else []
        else:
            print(f"Warning: summary CSV not found. Loading best model.")
            models_to_load = candidate_models[:1] if candidate_models else []

        print(f"Strategy '{self.ENSEMBLE_STRATEGY}' selected {len(models_to_load)} models: {models_to_load}")

        # Load selected models
        build_model = getattr(Training, "build_model")
        loaded, failed = [], []
        total = len(models_to_load)
        
        self.models = {}

        for idx, name in enumerate(models_to_load):
            progress = int((idx / total) * 100)
            print(f"Loading {name}... ({idx+1}/{total}) - {progress}%")
            
            path = os.path.join(self.WORK_DIR, f"{name}_best.pth")
            if not os.path.exists(path):
                print(f"Skipping {name}: File not found")
                failed.append(name)
                continue
                
            try:
                model, _ = build_model(name, num_classes=len(self.LABELS))
                state = torch.load(path, map_location=torch.device("cpu"))
                
                # Load checkpoint
                if isinstance(state, dict):
                    if "ema_state" in state and state["ema_state"]:
                        ema_state = state["ema_state"]
                        load_state = OrderedDict()
                        model_sd_template = model.state_dict()
                        ckpt_model_sd = state.get('model_state_dict', model_sd_template)
                        
                        for k, v_template in model_sd_template.items():
                            target_device = v_template.device
                            if v_template.dtype.is_floating_point and k in ema_state:
                                load_state[k] = ema_state[k].to(target_device)
                            elif k in ckpt_model_sd:
                                load_state[k] = ckpt_model_sd[k].to(target_device)
                            else:
                                load_state[k] = v_template
                        model.load_state_dict(load_state, strict=False)
                    elif "model_state_dict" in state:
                        model.load_state_dict(state["model_state_dict"], strict=False)
                    else:
                        model.load_state_dict(state, strict=False)
                else:
                    model.load_state_dict(state, strict=False)
                
                model.to(self.DEVICE)
                model.eval()
                self.models[name] = model
                loaded.append(name)
                print(f"✅ {name} loaded successfully")
                
                # Force garbage collection after each model
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                import traceback
                traceback.print_exc()
                failed.append(f"{name} ({str(e)[:30]})")
        
        # Calculate ensemble weights
        print("Calculating ensemble weights...")
        self.load_ensemble_weights()
        
        return loaded, failed
    
    def load_ensemble_weights(self):
        """Calculate QWK-based ensemble weights"""
        self.model_weights = {}
        models_to_use = list(self.models.keys())
        
        if not self.model_metrics or not models_to_use:
            if self.models:
                print("Fallback: Using equal weights.")
                self.model_weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
            return

        total_qwk = 0
        for model_name in models_to_use:
            total_qwk += self.model_metrics.get(model_name, {}).get('qwk', 0)
        
        if total_qwk > 0:
            for model_name in models_to_use:
                self.model_weights[model_name] = self.model_metrics.get(model_name, {}).get('qwk', 0) / total_qwk
        else:
            print("Fallback: QWK is 0 or missing. Using equal weights.")
            self.model_weights = {name: 1.0 / len(models_to_use) for name in models_to_use}
    
    def filter_models_by_strategy(self):
        """Filter models based on selected ensemble strategy"""
        if not self.model_metrics:
            return list(self.models.keys())
        
        all_metric_models = list(self.model_metrics.keys())
        
        if self.ENSEMBLE_STRATEGY == "all":
            return all_metric_models
        
        elif self.ENSEMBLE_STRATEGY == "top_performers":
            filtered = []
            for model_name in all_metric_models:
                metrics = self.model_metrics.get(model_name, {})
                qwk = metrics.get('qwk', 0)
                acc = metrics.get('accuracy', 0)
                if qwk >= self.MIN_QWK_THRESHOLD and acc >= self.MIN_ACCURACY_THRESHOLD:
                    filtered.append(model_name)
            
            if not filtered:
                sorted_models = sorted(self.model_metrics.items(), 
                                      key=lambda x: x[1]['qwk'], reverse=True)
                filtered = [m[0] for m in sorted_models[:3] if m[0] in all_metric_models]
            
            return filtered if filtered else all_metric_models
        
        elif self.ENSEMBLE_STRATEGY == "best_only":
            if not self.model_metrics:
                return list(self.models.keys())
            best_model = max(self.model_metrics.items(), 
                           key=lambda x: x[1].get('qwk', 0))[0]
            return [best_model] if best_model in all_metric_models else all_metric_models[:1]
        
        elif self.ENSEMBLE_STRATEGY == "adaptive":
            sorted_models = sorted(self.model_metrics.items(), 
                                  key=lambda x: x[1]['qwk'], reverse=True)
            n_models = max(1, len(sorted_models) // 2)
            return [m[0] for m in sorted_models[:n_models] if m[0] in all_metric_models]
        
        else:
            return all_metric_models
    
    def list_candidate_models(self):
        """List available model checkpoints"""
        names = []
        if os.path.exists(self.WORK_DIR):
            for fname in os.listdir(self.WORK_DIR):
                if fname.endswith("_best.pth"):
                    names.append(fname.replace("_best.pth", ""))
        
        if not names:
            names = getattr(Training, "DEFAULT_MODELS", 
                          ["ResNet50", "DenseNet201", "EfficientNetV2L"])
        return names
    
    def on_loading_progress(self, value, message):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{message} ({value}%)")
    
    def on_models_loaded(self, result):
        """Handle successful model loading"""
        loaded, failed = result
        
        self.loaded_models_list = loaded
        
        msg = ""
        if loaded:
            msg += f"✅ Successfully loaded {len(loaded)} models:\n"
            for name in loaded:
                weight = self.model_weights.get(name, 0)
                metrics = self.model_metrics.get(name, {})
                qwk = metrics.get('qwk', 0)
                msg += f"  • {name}\n"
                msg += f"    Weight: {weight:.3f} | QWK: {qwk:.4f}\n"
        
        if failed:
            msg += f"\n❌ Failed to load {len(failed)} models:\n"
            for name in failed:
                msg += f"  • {name}\n"
        
        self.model_status_text.setText(msg)
        self.log(msg)
        
        if loaded:
            self.models_loaded = True
            self.btn_load.setEnabled(True)
            self.btn_camera.setEnabled(True)
            self.btn_reload.setEnabled(True)
            self.status_indicator.set_status("success")
            self.status_label.setText(f"{len(loaded)} model(s) ready")
        else:
            self.status_indicator.set_status("error")
            self.status_label.setText("No models loaded")
            QMessageBox.critical(self, "Error", "No models could be loaded!")
    
    def on_models_load_error(self, error_tuple):
        """Handle model loading error"""
        self.log(f"❌ CRITICAL ERROR:\n{error_tuple[2]}")
        self.status_indicator.set_status("error")
        self.status_label.setText("Model loading failed")
        QMessageBox.critical(self, "Error", f"Model loading failed:\n{error_tuple[1]}")

    # ============================================================
    # IMAGE HANDLING
    # ============================================================
    
    def load_image(self):
        """Load image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Retina Image", "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        
        if file_path:
            self.display_image(file_path)
    
    def capture_camera(self):
        """Capture image from camera"""
        try:
            import cv2
        except ImportError:
            self.log("FATAL: OpenCV (cv2) not installed. Camera capture failed.")
            QMessageBox.critical(self, "Error", "OpenCV (cv2) is not installed.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, "Camera Error", "Cannot open camera")
            self.log("❌ Camera not accessible")
            return
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            os.makedirs(self.WORK_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tmp_path = os.path.join(self.WORK_DIR, f"camera_{timestamp}.jpg")
            cv2.imwrite(tmp_path, frame)
            self.display_image(tmp_path)
            self.log(f"📷 Camera image captured: {tmp_path}")
        else:
            QMessageBox.warning(self, "Capture Failed", "Failed to capture image")
    
    def display_image(self, file_path):
        """Display image in the UI"""
        try:
            self.current_image_path = file_path
            
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            file_size = os.path.getsize(file_path) / 1024
            img = Image.open(file_path)
            self.image_info_label.setText(
                f"📄 {os.path.basename(file_path)} | "
                f"Size: {img.size[0]}x{img.size[1]} | "
                f"{file_size:.1f} KB"
            )
            
            self.btn_test.setEnabled(self.models_loaded)
            
            self.diagnosis_label.setText("Ready for analysis...")
            self.confidence_label.setText("")
            self.clear_probability_bars()
            self.votes_table.setRowCount(0)
            
            self.log(f"📁 Image loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load image:\n{e}")
            self.log(f"❌ Failed to load image: {e}")

    # ============================================================
    # PREDICTION
    # ============================================================
    
    def run_test(self):
        """Start prediction in background thread"""
        if not self.models_loaded:
            QMessageBox.warning(self, "Not Ready", "Models are still loading...")
            return
        
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return
        
        normalized_path = os.path.normpath(self.current_image_path)
        is_outlier = normalized_path in self.outlier_paths_set
        
        if is_outlier:
            self.log("⚠️ WARNING: Image identified as a known outlier. Prediction may be unreliable.")
        
        self.set_controls_enabled(False)
        self.status_indicator.set_status("loading")
        self.status_label.setText("Analyzing...")
        
        self.log(f"\n{'='*60}")
        self.log(f"🔍 Starting analysis: {os.path.basename(self.current_image_path)}")
        self.log(f"{'='*60}")
        
        tta_steps = self.tta_spin.value()
        
        worker = Worker(self.run_prediction_logic, self.current_image_path, is_outlier=is_outlier, tta_steps=tta_steps)
        worker.signals.result.connect(self.on_prediction_complete)
        worker.signals.error.connect(self.on_prediction_error)
        worker.signals.finished.connect(self.on_prediction_finished)
        
        self.threadpool.start(worker)
    
    def run_prediction_logic(self, image_path: str, is_outlier: bool = False, tta_steps=1):
        """Run ensemble prediction with TTA and weighted voting"""
        import torch.nn.functional as F

        if not self.models:
            return None
        
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.val_transform(img).to(self.DEVICE)
        except Exception as e:
            raise RuntimeError(f"Image preprocessing failed: {e}")
        
        all_model_probs_dict = {}
        individual_results = {}
        
        with torch.no_grad():
            tta_tensors = _tta_variants(tensor.unsqueeze(0), tta_steps)
            
            for name, model in self.models.items():
                model_tta_probs = []
                
                for tta_img in tta_tensors:
                    logits = model(tta_img)
                    probs = F.softmax(logits, dim=1)
                    model_tta_probs.append(probs.cpu())
                
                mean_probs = torch.stack(model_tta_probs).mean(dim=0).numpy()
                all_model_probs_dict[name] = mean_probs
                
                pred_idx = int(np.argmax(mean_probs, axis=1)[0])
                confidence = float(mean_probs[0, pred_idx]) * 100.0
                individual_results[name] = (pred_idx, confidence)
        
        if not individual_results:
            raise RuntimeError("No predictions generated")
        
        models_used = list(self.models.keys())
        total_models = len(self.models)
        used_count = len(models_used)
        
        vote_counts = Counter([result[0] for result in individual_results.values()])
        
        if total_models == 1:
            consensus_type = f"single model [Strategy: {self.ENSEMBLE_STRATEGY}]"
        else:
            max_votes = max(vote_counts.values())
            if max_votes > total_models / 2:
                consensus_type = f"weighted ensemble with majority ({max_votes}/{total_models})"
            else:
                consensus_type = f"weighted ensemble ({used_count}/{total_models} models)"
            consensus_type += f" [Strategy: {self.ENSEMBLE_STRATEGY}]"
        
        # QWK-weighted ensemble
        weighted_probs = []
        total_weight = 0.0
        
        for model_name in models_used:
            if model_name in all_model_probs_dict:
                weight = self.model_weights.get(model_name, 1.0 / len(models_used))
                weighted_probs.append(all_model_probs_dict[model_name] * weight)
                total_weight += weight
        
        if not weighted_probs:
            raise RuntimeError("No models selected for ensemble")
        
        if total_weight > 0:
            ensemble_probs = np.sum(weighted_probs, axis=0) / total_weight
        else:
            ensemble_probs = np.mean([all_model_probs_dict[m] for m in models_used], axis=0)
        
        final_idx = int(np.argmax(ensemble_probs, axis=1)[0])
        final_conf = float(ensemble_probs[0, final_idx]) * 100.0
        
        return {
            'prediction': final_idx,
            'confidence': final_conf,
            'probabilities': ensemble_probs[0],
            'individual_results': individual_results,
            'models_used': models_used,
            'models_excluded': [],
            'consensus_type': consensus_type,
            'vote_counts': dict(vote_counts),
            'is_outlier': is_outlier
        }
    
    def on_prediction_complete(self, result):
        """Handle prediction results"""
        try:
            if result is None:
                QMessageBox.warning(self, "Error", "Prediction failed")
                return
            
            pred_idx = result['prediction']
            confidence = result['confidence']
            probs = result['probabilities']
            individual = result['individual_results']
            models_used = result['models_used']
            consensus = result['consensus_type']
            is_outlier = result.get('is_outlier', False)
            
            file_name = os.path.basename(self.current_image_path)
            id_code = os.path.splitext(file_name)[0]
            true_label = self.ground_truth_labels.get(id_code)
            
            class_name = self.CLASS_NAMES.get(pred_idx, f"Class {pred_idx}")
            color = self.CLASS_COLORS.get(pred_idx, "#6C757D")
            
            self.diagnosis_label.setText(f"🔬 {class_name}")
            self.diagnosis_label.setStyleSheet(
                f"font-size: 32px; font-weight: bold; color: {color}; "
                f"border: none; padding: 30px;"
            )
            
            self.confidence_label.setText(f"Confidence: {confidence:.1f}%")
            
            self.display_probability_bars(probs)
            self.display_model_votes(individual, models_used)
            
            self.log(f"\n{'─'*60}")
            self.log(f"📊 ENSEMBLE PREDICTION: {class_name} (Class {pred_idx})")
            self.log(f"   Confidence: {confidence:.2f}%")
            self.log(f"   Strategy: {consensus}")
            self.log(f"   Models used: {len(models_used)}/{len(self.models)}")
            
            popup_text = ""
            
            if true_label is not None:
                true_label = int(true_label)
                true_name = self.CLASS_NAMES.get(true_label, f"Class {true_label}")
                is_correct = (true_label == pred_idx)
                match = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                
                self.log(f"   Ground Truth: {true_name} (Class {true_label}) {match}")
                
                popup_text = f"<b>Ground Truth: {true_name} (Class {true_label})</b><br><br>"
                
                if not is_correct and is_outlier:
                    warning_msg = "⚠️ <b>Note:</b> This image is a known outlier, which may explain the incorrect prediction."
                    self.log(f"   {warning_msg}")
                    popup_text += f"<p style='color: {self.colors['warning']};'>{warning_msg}</p>"
                elif is_outlier:
                    warning_msg = "⚠️ <b>Note:</b> This image is a known outlier. Prediction reliability may be reduced."
                    self.log(f"   {warning_msg}")
                    popup_text += f"<p style='color: {self.colors['warning']};'>{warning_msg}</p>"
            else:
                self.log("   Ground Truth: N/A (not in CSV)")
                
                if is_outlier:
                    warning_msg = "⚠️ <b>Note:</b> This image is a known outlier. Prediction reliability may be reduced."
                    self.log(f"   {warning_msg}")
                    popup_text += f"<p style='color: {self.colors['warning']};'>{warning_msg}</p>"
            
            self.log(f"{'─'*60}\n")
            
            self.show_result_popup(class_name, confidence, true_label, pred_idx, popup_text)
            
            self.add_to_history(file_name, class_name, confidence, true_label)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.on_prediction_error((type(e), e, "Error in on_prediction_complete"))
    
    def on_prediction_error(self, error_tuple):
        """Handle prediction error"""
        self.log(f"\n❌ PREDICTION ERROR:\n{error_tuple[2]}")
        QMessageBox.critical(self, "Prediction Error", str(error_tuple[1]))
    
    def on_prediction_finished(self):
        """Re-enable controls after prediction"""
        self.set_controls_enabled(True)
        self.status_indicator.set_status("success")
        self.status_label.setText(f"{len(self.models)} model(s) ready")

    # ============================================================
    # UI UPDATES
    # ============================================================
    
    def display_probability_bars(self, probs):
        """Display probability distribution as bars"""
        self.clear_probability_bars()
        
        sorted_indices = np.argsort(probs)[::-1]
        
        for idx in sorted_indices:
            class_name = self.CLASS_NAMES[idx]
            prob = probs[idx]
            color = self.CLASS_COLORS[idx]
            
            bar = ProbabilityBar(class_name, prob, color)
            self.prob_layout.addWidget(bar)
    
    def clear_probability_bars(self):
        """Clear all probability bars"""
        while self.prob_layout.count():
            item = self.prob_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def display_model_votes(self, individual_results, models_used):
        """Display individual model predictions in table"""
        self.votes_table.setRowCount(len(individual_results))
        
        sorted_results = sorted(individual_results.items(), 
                               key=lambda x: x[1][1], reverse=True)
        
        for row, (name, (pred_idx, conf)) in enumerate(sorted_results):
            name_item = QTableWidgetItem(name)
            name_item.setFont(QFont("Arial", 10, QFont.Bold))
            self.votes_table.setItem(row, 0, name_item)
            
            pred_text = self.CLASS_NAMES[pred_idx]
            pred_item = QTableWidgetItem(pred_text)
            pred_item.setForeground(QColor(self.CLASS_COLORS[pred_idx]))
            pred_item.setFont(QFont("Arial", 10, QFont.Bold))
            self.votes_table.setItem(row, 1, pred_item)
            
            conf_item = QTableWidgetItem(f"{conf:.1f}%")
            self.votes_table.setItem(row, 2, conf_item)
            
            if name in models_used:
                used_text = "✓ Yes"
                used_item = QTableWidgetItem(used_text)
                used_item.setForeground(QColor(self.colors['success']))
            else:
                metrics = self.model_metrics.get(name, {})
                qwk = metrics.get('qwk', 0)
                used_text = f"✗ No (QWK: {qwk:.3f})"
                used_item = QTableWidgetItem(used_text)
                used_item.setForeground(QColor(self.colors['danger']))
            
            self.votes_table.setItem(row, 3, used_item)
        
        self.votes_table.resizeColumnsToContents()
    
    def show_result_popup(self, prediction, confidence, true_label, pred_idx, popup_text=""):
        """Show result in a popup dialog"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Diagnosis Complete")
        
        if not popup_text:
            popup_text = ""
        
        popup_text += f"<h2 style='color: {self.CLASS_COLORS.get(pred_idx, '#000')};'>"
        popup_text += f"🔬 {prediction}</h2>"
        popup_text += f"<p style='font-size: 14px;'><b>Confidence:</b> {confidence:.1f}%</p>"
        
        if true_label is not None and "Ground Truth:" not in popup_text:
            true_name = self.CLASS_NAMES.get(int(true_label), f"Class {true_label}")
            if int(true_label) == pred_idx:
                popup_text += f"<p style='color: green;'><b>✅ Matches Ground Truth:</b> {true_name}</p>"
            else:
                popup_text += f"<p style='color: red;'><b>❌ Ground Truth:</b> {true_name}</p>"
        
        popup_text += f"<hr><p style='font-size: 12px; color: #6C757D;'>"
        popup_text += f"{self.CLASS_DESCRIPTIONS.get(pred_idx, 'No description available.')}</p>"
        
        msg.setText(popup_text)
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
    def add_to_history(self, image_name, prediction, confidence, true_label):
        """Add prediction to history table"""
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_table.setItem(row, 0, QTableWidgetItem(timestamp))
        self.history_table.setItem(row, 1, QTableWidgetItem(image_name))
        self.history_table.setItem(row, 2, QTableWidgetItem(prediction))
        self.history_table.setItem(row, 3, QTableWidgetItem(f"{confidence:.1f}%"))
        
        if true_label is not None:
            gt_name = self.CLASS_NAMES.get(int(true_label), f"Class {true_label}")
            self.history_table.setItem(row, 4, QTableWidgetItem(gt_name))
        else:
            self.history_table.setItem(row, 4, QTableWidgetItem("N/A"))
        
        self.history_table.scrollToBottom()
    
    def clear_history(self):
        """Clear prediction history"""
        reply = QMessageBox.question(
            self, "Clear History",
            "Are you sure you want to clear all history?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.history_table.setRowCount(0)
            self.prediction_history = []
            self.log("🗑️ History cleared")

    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def log(self, message):
        """Add message to log"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def set_controls_enabled(self, enabled):
        """Enable/disable control buttons"""
        self.btn_load.setEnabled(enabled and self.models_loaded)
        self.btn_camera.setEnabled(enabled and self.models_loaded)
        self.btn_reload.setEnabled(enabled and self.models_loaded)
        has_image = self.current_image_path is not None and self.current_image_path != ""
        self.btn_test.setEnabled(enabled and self.models_loaded and has_image)
        self.strategy_combo.setEnabled(enabled)
        self.tta_spin.setEnabled(enabled)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    app = QApplication(sys.argv)
    
    app.setApplicationName("DR CAD System")
    app.setOrganizationName("Medical AI")
    
    window = ModernDRCAD()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()