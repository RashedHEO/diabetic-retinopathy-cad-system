import sys, os, numpy as np, cv2, torch, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict, Counter
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QTextEdit, QMessageBox, QHBoxLayout, QProgressBar,
    QFrame, QScrollArea, QGridLayout, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QGroupBox, QSizePolicy
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
        
        # Background
        painter.fillRect(self.rect(), QColor("#F8F9FA"))
        
        # Probability bar
        bar_width = int(self.width() * 0.7 * self.probability)
        painter.fillRect(10, 10, bar_width, 20, QColor(self.color))
        
        # Border
        painter.setPen(QPen(QColor("#DEE2E6"), 1))
        painter.drawRect(10, 10, int(self.width() * 0.7), 20)
        
        # Text
        painter.setPen(QColor("#212529"))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(15, 25, self.class_name)
        
        # Percentage
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
            
            # Divider
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
        self.status = "idle"  # idle, loading, success, error
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
        self.setStyleSheet(f"""
            background-color: {color};
            border-radius: 6px;
        """)


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
        
        # Parameters from training
        # (Line 187)

        # Parameters from training
        HOME = os.path.expanduser("~")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.WORK_DIR = os.path.join(BASE_DIR, "Work") # تأكد أن مجلد Work بجانب ملف الكود
        self.SUMMARY_CSV_PATH = os.path.join(self.WORK_DIR, "summary_metrics.csv")
    
        self.CSV_LABELS_PATH = os.path.join(BASE_DIR, "archive_data", "trainLabels_cropped_2Lines.csv")
        self.OUTLIER_FILE_PATH = os.path.join(BASE_DIR, "EDA_Results_DeepLearning", "outliers.csv")
        
        # Ensemble strategy selection
        self.ENSEMBLE_STRATEGY = "all" # Options: "all", "top_performers", "best_only", "adaptive"
        self.MIN_QWK_THRESHOLD = 0.75  # Only use models with QWK >= 0.75 for "top_performers"
        self.MIN_ACCURACY_THRESHOLD = 0.80  # Only use models with accuracy >= 80%
        
        self.LABELS = [0, 1, 2, 3, 4]
        self.CLASS_NAMES = {
            0: "No DR", 
            1: "Mild NPDR", 
            2: "Moderate NPDR", 
            3: "Severe NPDR", 
            4: "Proliferative DR"
        }
        self.CLASS_COLORS = {
            0: "#28A745",  # Green
            1: "#FFC107",  # Yellow
            2: "#FD7E14",  # Orange
            3: "#DC3545",  # Red
            4: "#6F42C1"   # Purple
        }
        
        # NEW: Class descriptions for better user experience
        self.CLASS_DESCRIPTIONS = {
            0: "No signs of diabetic retinopathy detected. Regular monitoring recommended.",
            1: "Mild non-proliferative diabetic retinopathy. Early stage with microaneurysms.",
            2: "Moderate non-proliferative diabetic retinopathy. More widespread retinal damage.",
            3: "Severe non-proliferative diabetic retinopathy. Extensive retinal damage requiring close monitoring.",
            4: "Proliferative diabetic retinopathy. Advanced stage with new blood vessel growth."
        }
        
        self.IMG_SIZE = getattr(Training, "DEFAULT_IMG_SIZE", 224)
        self.DEVICE = getattr(Training, "DEVICE", 
                              torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        _, self.val_transform = getattr(Training, "get_transforms")(self.IMG_SIZE)
        
        self.models = {}
        self.model_weights = {}
        self.model_metrics = {}  # Store QWK, accuracy, F1 for each model
        self.ground_truth_labels = {}
        self.outlier_paths_set = set()
        self.current_image_path = None
        self.models_loaded = False
        self.prediction_history = []
        
        self.threadpool = QThreadPool()
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
            QTabWidget::pane {{
                border: 1px solid #DEE2E6;
                border-radius: 8px;
                background-color: white;
            }}
            QTabBar::tab {{
                background-color: #E9ECEF;
                border: 1px solid #DEE2E6;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 10px 20px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: white;
                border-bottom: 2px solid {self.colors['primary']};
            }}
            QTableWidget {{
                background-color: white;
                border: 1px solid #DEE2E6;
                border-radius: 8px;
                gridline-color: #DEE2E6;
            }}
            QHeaderView::section {{
                background-color: #F8F9FA;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #DEE2E6;
                font-weight: bold;
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
        
        # Progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Image and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Results and details
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
        
        # Title
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
        
        # Device info
        device_label = QLabel(f"Device: {str(self.DEVICE).upper()}")
        device_label.setStyleSheet("color: white; background: transparent; font-size: 12px;")
        layout.addWidget(device_label)
        
        header.setLayout(layout)
        header.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return header
        
    def create_left_panel(self):
        """Create left panel with image and controls"""
        container = ModernCard()
        
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
        self.image_label.setStyleSheet("""
            color: #6C757D;
            font-size: 16px;
            background: transparent;
            border: none;
        """)
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
                df = pd.read_csv(self.OUTLIER_FILE_PATH)
                if 'path' in df.columns:
                    # Use os.path.normpath for consistent path matching (Windows)
                    self.outlier_paths_set = set(os.path.normpath(p) for p in df['path'].tolist())
                    self.log(f"✅ Loaded {len(self.outlier_paths_set)} outliers for pre-check.")
                else:
                    self.log("⚠️ Outlier CSV found but missing 'path' column.")
            except Exception as e:
                self.log(f"⚠️ Warning: Could not load outlier CSV. {e}")
        else:
            self.log(f"ℹ️ Outlier file not found at {self.OUTLIER_FILE_PATH}. Skipping outlier pre-check.")
    
    def start_model_loading(self):
        """Start background model loading"""
        self.status_indicator.set_status("loading")
        self.status_label.setText("Loading models...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setFormat("Loading models, please wait...")
        
        worker = Worker(self.load_models_with_progress)
        worker.signals.result.connect(self.on_models_loaded)
        worker.signals.error.connect(self.on_models_load_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        
        self.threadpool.start(worker)
    
    def load_models_with_progress(self):
        """Load models and return results (no signals here - handled by Worker)"""
        build_model = getattr(Training, "build_model")
        model_names = self.list_candidate_models()
        
        loaded, failed = [], []
        total = len(model_names)
        
        for idx, name in enumerate(model_names):
            print(f"Loading {name}... ({idx+1}/{total})")  # Console feedback
            
            path = os.path.join(self.WORK_DIR, f"{name}_best.pth")
            if not os.path.exists(path):
                failed.append(name)
                continue
                
            try:
                res = build_model(name, num_classes=len(self.LABELS))
                model = res[0] if isinstance(res, tuple) else res
                state = torch.load(path, map_location=self.DEVICE, weights_only=False)
                
                # Load checkpoint
                if isinstance(state, dict):
                    if "ema_state" in state and state["ema_state"]:
                        ema_state = state["ema_state"]
                        load_state = OrderedDict()
                        model_sd_template = model.state_dict()
                        ckpt_model_sd = state.get('model_state_dict', model_sd_template)
                        
                        for k, v_template in model_sd_template.items():
                            if v_template.dtype.is_floating_point and k in ema_state:
                                load_state[k] = ema_state[k].to(v_template.device)
                            elif k in ckpt_model_sd:
                                load_state[k] = ckpt_model_sd[k].to(v_template.device)
                            else:
                                load_state[k] = v_template
                        model.load_state_dict(load_state, strict=False)
                    elif "model_state_dict" in state:
                        model.load_state_dict(state["model_state_dict"], strict=False)
                    else:
                        model.load_state_dict(state, strict=False)
                
                model.to(self.DEVICE)
                model.eval()
                self.models[name] = model
                loaded.append(name)
                print(f"✅ {name} loaded successfully")
                
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                failed.append(f"{name} ({str(e)[:30]})")
        
        # Load ensemble weights
        print("Loading ensemble weights...")
        self.load_ensemble_weights()
        
        return loaded, failed
    
    def load_ensemble_weights(self):
        """Load QWK-based ensemble weights with smart filtering"""
        self.model_weights = {}
        self.model_metrics = {}  # Store full metrics for filtering
        
        if os.path.exists(self.SUMMARY_CSV_PATH):
            try:
                df = pd.read_csv(self.SUMMARY_CSV_PATH)
                df = df[df['model'] != 'Ensemble']
                
                # Store metrics for each model
                for _, row in df.iterrows():
                    if row['model'] in self.models:
                        self.model_metrics[row['model']] = {
                            'qwk': row.get('qwk', 0),
                            'accuracy': row.get('accuracy', 0),
                            'f1': row.get('f1_weighted', 0)
                        }
                
                # Filter models based on strategy
                filtered_models = self.filter_models_by_strategy()
                
                # Calculate weights only for filtered models
                if filtered_models:
                    total_qwk = sum(self.model_metrics[m]['qwk'] for m in filtered_models)
                    if total_qwk > 0:
                        for model_name in filtered_models:
                            self.model_weights[model_name] = self.model_metrics[model_name]['qwk'] / total_qwk
                        return
                    
            except Exception as e:
                self.log(f"⚠️ Error loading weights: {e}")
        
        # Fallback to equal weights for all models
        if self.models:
            self.model_weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
    
    def filter_models_by_strategy(self):
        """Filter models based on selected ensemble strategy"""
        if not self.model_metrics:
            return list(self.models.keys())
        
        if self.ENSEMBLE_STRATEGY == "all":
            # Use all models
            return list(self.models.keys())
        
        elif self.ENSEMBLE_STRATEGY == "top_performers":
            # Only use models above QWK and accuracy thresholds
            filtered = []
            for model_name in self.models.keys():
                metrics = self.model_metrics.get(model_name, {})
                qwk = metrics.get('qwk', 0)
                acc = metrics.get('accuracy', 0)
                if qwk >= self.MIN_QWK_THRESHOLD and acc >= self.MIN_ACCURACY_THRESHOLD:
                    filtered.append(model_name)
            
            if not filtered:  # If too strict, fall back to top 3 by QWK
                sorted_models = sorted(self.model_metrics.items(), 
                                      key=lambda x: x[1]['qwk'], reverse=True)
                filtered = [m[0] for m in sorted_models[:3] if m[0] in self.models]
            
            return filtered if filtered else list(self.models.keys())
        
        elif self.ENSEMBLE_STRATEGY == "best_only":
            # Use only the single best model by QWK
            best_model = max(self.model_metrics.items(), 
                           key=lambda x: x[1]['qwk'])[0]
            return [best_model] if best_model in self.models else list(self.models.keys())
        
        elif self.ENSEMBLE_STRATEGY == "adaptive":
            # Use top 50% of models by QWK
            sorted_models = sorted(self.model_metrics.items(), 
                                  key=lambda x: x[1]['qwk'], reverse=True)
            n_models = max(1, len(sorted_models) // 2)
            return [m[0] for m in sorted_models[:n_models] if m[0] in self.models]
        
        else:
            return list(self.models.keys())
    
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
        
        msg = ""
        if loaded:
            msg += f"✅ Successfully loaded {len(loaded)} models:\n"
            for name in loaded:
                weight = self.model_weights.get(name, 0)
                msg += f"  • {name} (weight: {weight:.3f})\n"
        
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
            self.status_indicator.set_status("success")
            self.status_label.setText(f"{len(loaded)} models ready")
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
            
            # Load and display image
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            # Show image info
            file_size = os.path.getsize(file_path) / 1024  # KB
            img = Image.open(file_path)
            self.image_info_label.setText(
                f"📄 {os.path.basename(file_path)} | "
                f"Size: {img.size[0]}x{img.size[1]} | "
                f"{file_size:.1f} KB"
            )
            
            # Enable test button
            self.btn_test.setEnabled(self.models_loaded)
            
            # Clear previous results
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
        """Start prediction in background thread with outlier detection"""
        if not self.models_loaded:
            QMessageBox.warning(self, "Not Ready", "Models are still loading...")
            return
        
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Please load an image first")
            return
        
        # NEW: Check if the current image is an outlier
        normalized_path = os.path.normpath(self.current_image_path)
        is_outlier = normalized_path in self.outlier_paths_set
        
        if is_outlier:
            self.log("⚠️ WARNING: Image identified as a known outlier. Prediction may be unreliable.")
        
        # Disable controls
        self.set_controls_enabled(False)
        self.status_indicator.set_status("loading")
        self.status_label.setText("Analyzing...")
        
        self.log(f"\n{'='*60}")
        self.log(f"🔍 Starting analysis: {os.path.basename(self.current_image_path)}")
        self.log(f"{'='*60}")
        
        # NEW: Pass is_outlier flag to prediction worker
        worker = Worker(self.run_prediction_logic, self.current_image_path, is_outlier=is_outlier)
        worker.signals.result.connect(self.on_prediction_complete)
        worker.signals.error.connect(self.on_prediction_error)
        worker.signals.finished.connect(self.on_prediction_finished)
        
        self.threadpool.start(worker)
    
    def run_prediction_logic(self, image_path: str, is_outlier: bool = False, tta_steps=8):
        """
        Run ensemble prediction with TTA and weighted voting.
        Executed in worker thread.
        """
        if not self.models:
            return None
        
        # Load and transform image
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self.val_transform(img).to(self.DEVICE)
        except Exception as e:
            raise RuntimeError(f"Image preprocessing failed: {e}")
        
        all_model_probs_dict = {}
        individual_results = {}
        
        # Run each model with TTA
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(self.DEVICE.type == "cuda")):
            tta_tensors = _tta_variants(tensor.unsqueeze(0), tta_steps)
            
            for name, model in self.models.items():
                model_tta_probs = []
                
                for tta_img in tta_tensors:
                    logits = model(tta_img)
                    probs = F.softmax(logits, dim=1)
                    model_tta_probs.append(probs.cpu())
                
                # Average TTA predictions
                mean_probs = torch.stack(model_tta_probs).mean(dim=0).numpy()
                all_model_probs_dict[name] = mean_probs
                
                pred_idx = int(np.argmax(mean_probs, axis=1)[0])
                confidence = float(mean_probs[0, pred_idx]) * 100.0
                individual_results[name] = (pred_idx, confidence)
        
        if not individual_results:
            raise RuntimeError("No predictions generated")
        
        # === SMART MODEL SELECTION ===
        available_models = list(self.models.keys())
        models_to_use = [m for m in available_models if m in self.model_weights]
        
        if not models_to_use:
            models_to_use = available_models
        
        models_used = models_to_use
        total_models = len(self.models)
        used_count = len(models_used)
        
        # Count votes for information
        vote_counts = Counter([result[0] for result in individual_results.values()])
        max_votes = max(vote_counts.values())
        
        # Consensus info
        if max_votes > total_models / 2:
            consensus_type = f"weighted ensemble with majority ({max_votes}/{total_models})"
        else:
            consensus_type = f"weighted ensemble ({used_count}/{total_models} models)"
        
        # Add strategy info
        consensus_type += f" [Strategy: {self.ENSEMBLE_STRATEGY}]"
        
        # === QWK-WEIGHTED ENSEMBLE (only selected models) ===
        weighted_probs = []
        total_weight = 0.0
        
        for model_name in models_used:
            if model_name in all_model_probs_dict:
                weight = self.model_weights.get(model_name, 0.0)
                weighted_probs.append(all_model_probs_dict[model_name] * weight)
                total_weight += weight
        
        if not weighted_probs:
            raise RuntimeError("No models selected for ensemble")
        
        # Normalize ensemble probabilities
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
            'models_excluded': [m for m in available_models if m not in models_used],
            'consensus_type': consensus_type,
            'vote_counts': dict(vote_counts),
            'is_outlier': is_outlier  # NEW: Include outlier flag in results
        }
    
    def on_prediction_complete(self, result):
        """Handle prediction results with outlier detection"""
        try:
            if result is None:
                QMessageBox.warning(self, "Error", "Prediction failed")
                return
            
            pred_idx = result['prediction']
            confidence = result['confidence']
            probs = result['probabilities']
            individual = result['individual_results']
            models_used = result['models_used']
            models_excluded = result.get('models_excluded', [])
            consensus = result['consensus_type']
            vote_counts = result['vote_counts']
            is_outlier = result.get('is_outlier', False)  # NEW: Get outlier flag
            
            # Get ground truth
            file_name = os.path.basename(self.current_image_path)
            id_code = os.path.splitext(file_name)[0]
            true_label = self.ground_truth_labels.get(id_code)
            
            # Update diagnosis display
            class_name = self.CLASS_NAMES.get(pred_idx, f"Class {pred_idx}")
            color = self.CLASS_COLORS.get(pred_idx, "#6C757D")
            
            self.diagnosis_label.setText(f"🔬 {class_name}")
            self.diagnosis_label.setStyleSheet(
                f"font-size: 32px; font-weight: bold; color: {color}; "
                f"border: none; padding: 30px;"
            )
            
            self.confidence_label.setText(f"Confidence: {confidence:.1f}%")
            
            # Display probability bars and model votes
            self.display_probability_bars(probs)
            self.display_model_votes(individual, models_used)
            
            # Log detailed results
            self.log(f"\n{'─'*60}")
            self.log(f"📊 ENSEMBLE PREDICTION: {class_name} (Class {pred_idx})")
            self.log(f"   Confidence: {confidence:.2f}%")
            self.log(f"   Strategy: {consensus}")
            self.log(f"   Models used: {len(models_used)}/{len(self.models)}")
            
            if models_excluded:
                self.log(f"   ⚠️ Excluded models: {', '.join(models_excluded)}")
                for excluded in models_excluded:
                    metrics = self.model_metrics.get(excluded, {})
                    self.log(f"      {excluded}: QWK={metrics.get('qwk', 0):.4f}, ACC={metrics.get('accuracy', 0):.4f}")
            
            # NEW: Build popup text with outlier information
            popup_text = ""
            
            # Handle ground truth comparison with outlier detection
            if true_label is not None:
                true_label = int(true_label)
                true_name = self.CLASS_NAMES.get(true_label, f"Class {true_label}")
                is_correct = (true_label == pred_idx)
                match = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                
                self.log(f"   Ground Truth: {true_name} (Class {true_label}) {match}")
                
                # Build popup text
                popup_text = f"<b>Ground Truth: {true_name} (Class {true_label})</b><br><br>"
                
                # NEW: Special handling for wrong predictions on outliers
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
                
                # Even without ground truth, warn about outliers
                if is_outlier:
                    warning_msg = "⚠️ <b>Note:</b> This image is a known outlier. Prediction reliability may be reduced."
                    self.log(f"   {warning_msg}")
                    popup_text += f"<p style='color: {self.colors['warning']};'>{warning_msg}</p>"
            
            self.log(f"{'─'*60}\n")
            
            # Show result popup with outlier info
            self.show_result_popup(class_name, confidence, true_label, pred_idx, popup_text)
            
            # Add to history
            self.add_to_history(
                file_name, 
                class_name, 
                confidence, 
                true_label
            )
            
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
        self.status_label.setText(f"{len(self.models)} models ready")

    # ============================================================
    # UI UPDATES
    # ============================================================
    
    def display_probability_bars(self, probs):
        """Display probability distribution as bars"""
        self.clear_probability_bars()
        
        # Sort by probability (descending)
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
        
        # Sort by confidence (descending)
        sorted_results = sorted(individual_results.items(), 
                               key=lambda x: x[1][1], reverse=True)
        
        for row, (name, (pred_idx, conf)) in enumerate(sorted_results):
            # Model name
            name_item = QTableWidgetItem(name)
            name_item.setFont(QFont("Arial", 10, QFont.Bold))
            self.votes_table.setItem(row, 0, name_item)
            
            # Prediction
            pred_text = self.CLASS_NAMES[pred_idx]
            pred_item = QTableWidgetItem(pred_text)
            pred_item.setForeground(QColor(self.CLASS_COLORS[pred_idx]))
            pred_item.setFont(QFont("Arial", 10, QFont.Bold))
            self.votes_table.setItem(row, 1, pred_item)
            
            # Confidence
            conf_item = QTableWidgetItem(f"{conf:.1f}%")
            self.votes_table.setItem(row, 2, conf_item)
            
            # Used in ensemble
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
        """Show result in a popup dialog with outlier information"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Diagnosis Complete")
        
        # Build popup content
        if not popup_text:
            popup_text = ""
        
        # Add prediction header
        popup_text += f"<h2 style='color: {self.CLASS_COLORS.get(pred_idx, '#000')};'>"
        popup_text += f"🔬 {prediction}</h2>"
        popup_text += f"<p style='font-size: 14px;'><b>Confidence:</b> {confidence:.1f}%</p>"
        
        # Add ground truth comparison if available and not already in popup_text
        if true_label is not None and "Ground Truth:" not in popup_text:
            true_name = self.CLASS_NAMES.get(int(true_label), f"Class {true_label}")
            if int(true_label) == pred_idx:
                popup_text += f"<p style='color: green;'><b>✅ Matches Ground Truth:</b> {true_name}</p>"
            else:
                popup_text += f"<p style='color: red;'><b>❌ Ground Truth:</b> {true_name}</p>"
        
        # Add clinical description
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
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_table.setItem(row, 0, QTableWidgetItem(timestamp))
        
        # Image name
        self.history_table.setItem(row, 1, QTableWidgetItem(image_name))
        
        # Prediction
        self.history_table.setItem(row, 2, QTableWidgetItem(prediction))
        
        # Confidence
        self.history_table.setItem(row, 3, QTableWidgetItem(f"{confidence:.1f}%"))
        
        # Ground truth
        if true_label is not None:
            gt_name = self.CLASS_NAMES.get(int(true_label), f"Class {true_label}")
            self.history_table.setItem(row, 4, QTableWidgetItem(gt_name))
        else:
            self.history_table.setItem(row, 4, QTableWidgetItem("N/A"))
        
        # Scroll to bottom
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
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def set_controls_enabled(self, enabled):
        """Enable/disable control buttons"""
        self.btn_load.setEnabled(enabled and self.models_loaded)
        self.btn_camera.setEnabled(enabled and self.models_loaded)
        # Test button needs both models loaded AND an image present
        has_image = self.current_image_path is not None and self.current_image_path != ""
        self.btn_test.setEnabled(enabled and self.models_loaded and has_image)
    
    def get_class_index(self, class_name):
        """Get class index from name"""
        for idx, name in self.CLASS_NAMES.items():
            if name == class_name:
                return idx
        return 0


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("DR CAD System")
    app.setOrganizationName("Medical AI")
    
    # Create and show main window
    window = ModernDRCAD()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()