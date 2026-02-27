import asyncio
import pyaudio
import os
import threading
from google import genai
from google.genai import types
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
                               QPushButton, QLabel, QSlider, QComboBox, QTextEdit,
                               QGroupBox, QSpinBox, QCheckBox, QFrame, QGridLayout)
from PySide6.QtCore import QThread, Signal, QTimer, Qt, QPropertyAnimation, QEasingCurve, QSize, Property, QParallelAnimationGroup
from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush, QLinearGradient
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QScrollArea, QGraphicsOpacityEffect
import sys
import math
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModernKnob(QWidget):
    valueChanged = Signal(int)
    
    def __init__(self, min_val=0, max_val=100, initial_val=50):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.setFixedSize(40, 40)
        self.angle = 0
        self.dragging = False
        self.setAttribute(Qt.WA_TranslucentBackground)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        
        pen = QPen(QColor(180, 180, 180, 200), 2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(240, 240, 240, 150)))
        painter.drawEllipse(5, 5, 30, 30)
        
        painter.setBrush(QBrush(QColor(220, 220, 220, 180)))
        painter.drawEllipse(10, 10, 20, 20)
        
        painter.setPen(QPen(QColor(255, 140, 0), 3))
        angle_rad = (self.value - self.min_val) / (self.max_val - self.min_val) * 270 - 135
        angle_rad = angle_rad * math.pi / 180
        x = 20 + 10 * math.cos(angle_rad)
        y = 20 + 10 * math.sin(angle_rad)
        painter.drawLine(20, 20, int(x), int(y))
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            
    def mouseReleaseEvent(self, event):
        self.dragging = False
        
    def mouseMoveEvent(self, event):
        if self.dragging:
            dx = event.x() - 20
            dy = event.y() - 20
            angle = math.atan2(dy, dx) * 180 / math.pi
            angle = (angle + 135) % 360
            if angle > 270:
                angle = 270
            self.value = int(self.min_val + (angle / 270) * (self.max_val - self.min_val))
            self.valueChanged.emit(self.value)
            self.update()

class VerticalSlider(QFrame):
    valueChanged = Signal(float)
    deleteRequested = Signal()
    
    def __init__(self, label="", initial_value=1.0):
        super().__init__()
        self.label = label
        self.value = initial_value
        self.setFixedWidth(80)
        self.setFixedHeight(300)
        self._fade_anim = None
        
        # Set up opacity effect for fade-in animation
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0.0)
        
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(250,250,250,0);
                border: none;
                border-radius: 8px;
                margin: 2px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)
        
        label_widget = QLabel(self.label[:10])
        label_widget.setAlignment(Qt.AlignCenter)
        label_widget.setStyleSheet("""
            QLabel {
                font-weight: bold;
                font-size: 10px;
                color: #333;
                background-color: transparent;
                padding: 1px;
            }
        """)
        layout.addWidget(label_widget)
        
        self.qt_slider = QSlider(Qt.Vertical)
        self.qt_slider.setMinimum(0)
        self.qt_slider.setMaximum(100)
        self.qt_slider.setFixedHeight(200)
        self.qt_slider.setValue(int(initial_value * 100))
        self.qt_slider.setStyleSheet("""
            QSlider::groove:vertical {
                border: none;
                width: 8px;
                background: transparent;
                border-radius: 4px;
            }
            QSlider::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8C00, stop:1 #FFA500);
                border: 2px solid #CC7000;
                width: 20px;
                height: 20px;
                border-radius: 12px;
                margin: -8px 0;
            }
            QSlider::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FFA500, stop:1 #FFB520);
            }
            QSlider::handle:vertical:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #E67300, stop:1 #FF8C00);
            }
            QSlider::add-page:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #FF8C00, stop:1 #FFA500);
                border: none;
                border-radius: 4px;
            }
            QSlider::sub-page:vertical {
                background: transparent;
                border: none;
                border-radius: 4px;
            }
        """)
        self.qt_slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.qt_slider, 1, Qt.AlignCenter)
        
        self.value_label = QLabel(f"{self.value:.2f}")
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                font-weight: bold;
                color: #333;
                background-color: rgba(255, 255, 255, 200);
                border: 1px solid #DDD;
                border-radius: 3px;
                padding: 2px;
                min-height: 14px;
                max-height: 18px;
            }
        """)
        layout.addWidget(self.value_label)
        
        self.delete_btn = QPushButton("✗")
        self.delete_btn.setFixedSize(25, 22)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF4444;
                color: white;
                border: none;
                border-radius: 11px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF2222;
            }
            QPushButton:pressed {
                background-color: #DD2222;
            }
        """)
        self.delete_btn.clicked.connect(self.deleteRequested.emit)
        layout.addWidget(self.delete_btn, 1, Qt.AlignCenter)
        
        # Ensure proper geometry calculation
        self.updateGeometry()
        
    def on_slider_changed(self, qt_value):
        self.value = qt_value / 100.0
        self.value_label.setText(f"{self.value:.2f}")
        self.valueChanged.emit(self.value)
    
    def fade_in(self):
        """Animate opacity from 0 to 1"""
        if not self.opacity_effect:
            return
            
        self._fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self._fade_anim.setDuration(600)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)
        
        # When animation finishes, remove the effect (we don't need it anymore)
        def cleanup_effect():
            if self.opacity_effect:
                self.setGraphicsEffect(None)
                self.opacity_effect = None
        
        self._fade_anim.finished.connect(cleanup_effect)
        self._fade_anim.start()

class SemiOpaqueArea(QFrame):
    """Semi-opaque placeholder area that slides to the right as sliders are added"""
    
    def __init__(self, initial_width):
        super().__init__()
        self._area_width = initial_width
        self._width_anim = None
        self.setFixedHeight(408)  # Fill almost all vertical space
        self.updateWidth()
        
        self.setStyleSheet("""
            SemiOpaqueArea {
                background-color: rgba(250, 250, 250, 255);
                border-radius: 12px;
            }
        """)
        
        # Add a label in the center
        layout = QVBoxLayout(self)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                color: rgba(100, 100, 120, 180);
                font-size: 16px;
                font-weight: bold;
                font-style: italic;
                background-color: transparent;
                border: none;
            }
        """)
        layout.addWidget(self.label)
        
    def updateWidth(self):
        self.setFixedWidth(max(100, self._area_width))
        
    def animate_resize(self, new_width):
        """Animate area width change"""
        self._width_anim = QPropertyAnimation(self, b"area_width")
        self._width_anim.setDuration(500)
        self._width_anim.setStartValue(self._area_width)
        self._width_anim.setEndValue(max(100, new_width))
        self._width_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._width_anim.start()
    
    def get_area_width(self):
        return self._area_width
    
    def set_area_width(self, width):
        self._area_width = width
        self.updateWidth()
    
    area_width = Property(int, get_area_width, set_area_width)

class LyriaAudioThread(QThread):
    status_update = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.session = None
        self.client = None
        self.p = None
        self.loop = None
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        self.BUFFER_SECONDS = 1
        self.CHUNK = 4200
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.MODEL = 'models/lyria-realtime-exp'
        self.OUTPUT_RATE = 48000
        
        self.config = types.LiveMusicGenerationConfig()
        self.config.bpm = 120
        self.config.scale = types.Scale.A_FLAT_MAJOR_F_MINOR
        
    def run(self):
        if not self.api_key:
            self.error_occurred.emit("Google API Key not found. Please set GOOGLE_API_KEY in your .env file")
            return
            
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.p = pyaudio.PyAudio()
            self.client = genai.Client(
                api_key=self.api_key,
                http_options={'api_version': 'v1alpha'}
            )
            
            self.running = True
            self.status_update.emit("Connecting to Lyria...")
            self.loop.run_until_complete(self.async_main())
            
        except Exception as e:
            self.error_occurred.emit(f"Error in audio thread: {str(e)}")
        finally:
            self.cleanup()
    
    async def async_main(self):
        try:
            async with self.client.aio.live.music.connect(model=self.MODEL) as session:
                self.session = session
                await session.set_music_generation_config(config=self.config)
                await session.set_weighted_prompts(
                    prompts=[types.WeightedPrompt(text="Piano", weight=1.0)]
                )
                self.status_update.emit("Connected! Starting music generation...")
                await session.play()
                await self.receive_audio()
        except Exception as e:
            self.error_occurred.emit(f"Session error: {str(e)}")
    
    async def receive_audio(self):
        try:
            chunks_count = 0
            output_stream = self.p.open(
                format=self.FORMAT, 
                channels=self.CHANNELS, 
                rate=self.OUTPUT_RATE, 
                output=True, 
                frames_per_buffer=self.CHUNK
            )
            
            async for message in self.session.receive():
                if not self.running:
                    break
                    
                chunks_count += 1
                if chunks_count == 1:
                    await asyncio.sleep(self.BUFFER_SECONDS)
                
                if message.server_content:
                    audio_data = message.server_content.audio_chunks[0].data
                    output_stream.write(audio_data)
                elif message.filtered_prompt:
                    self.status_update.emit(f"Prompt filtered: {message.filtered_prompt}")
                
                await asyncio.sleep(10**-12)
                
        except Exception as e:
            self.error_occurred.emit(f"Audio receive error: {str(e)}")
    
    def stop_audio(self):
        self.running = False
        if self.loop and self.session:
            asyncio.run_coroutine_threadsafe(self.session.stop(), self.loop)
    
    def cleanup(self):
        if self.p:
            self.p.terminate()
        if self.loop:
            self.loop.close()

class LyriaMusicGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.audio_thread = None
        self.sliders = []
        self.weighted_prompts = []
        self.semi_opaque_area = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("LYRIA-6")
        self.setFixedWidth(800)
        self.setFixedHeight(850)
        
        self.setStyleSheet("""
            LyriaMusicGenerator {
                background-color: rgba(232, 232, 232, 0);
                border: 2px solid #000000;
                border-radius: 15px;
            }
            QWidget {
                color: #333;
                font-family: Arial, sans-serif;
            }
            QPushButton {
                background-color: rgba(208, 208, 208, 180);
                border: 1px solid #AAA;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(192, 192, 192, 200);
            }
            QPushButton:pressed {
                background-color: rgba(176, 176, 176, 200);
            }
            QPushButton:checked {
                background-color: #FF8C00;
                color: white;
            }
            QLineEdit {
                background-color: rgba(255, 255, 255, 255);
                border: 1px solid #CCC;
                border-radius: 4px;
                padding: 6px;
                font-size: 12px;
            }
            QLabel {
                font-size: 11px;
                color: #555;
                background-color: transparent;
            }
            QComboBox {
                background-color: rgba(255, 255, 255, 255);
                border: 1px solid #CCC;
                border-radius: 4px;
                padding: 4px;
                font-size: 11px;
            }
            QFrame {
                background-color: rgba(245, 245, 245, 255);
                border-radius: 15px;
            }
            QScrollArea {
                background-color: transparent;
                border: none;
            }
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(245, 245, 245, 255);
                border-radius: 15px;
                padding: 10px;
            }
        """)
        header_layout = QHBoxLayout(header_frame)
        
        title_label = QLabel("LYRIA-6")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setStyleSheet("color: #666; background-color: transparent;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        self.status_display = QLabel("READY")
        self.status_display.setStyleSheet("""
            QLabel {
                background-color: rgba(34, 34, 34, 200);
                color: #0F0;
                padding: 8px 16px;
                border-radius: 15px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        header_layout.addWidget(self.status_display)
        main_layout.addWidget(header_frame)
        
        # Control section
        control_frame = QFrame()
        control_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(222, 222, 222, 150);
                border: 2px solid #000000;
                border-radius: 15px;
                padding: 15px;
            }
        """)
        control_layout = QVBoxLayout(control_frame)
        
        top_control_layout = QHBoxLayout()
        knobs_layout = QHBoxLayout()
        
        # BPM Knob
        bpm_container = QVBoxLayout()
        self.bpm_knob = ModernKnob(60, 200, 120)
        self.bpm_knob.valueChanged.connect(self.update_bpm)
        bpm_container.addWidget(self.bpm_knob)
        bpm_label = QLabel("BPM")
        bpm_label.setAlignment(Qt.AlignCenter)
        bpm_container.addWidget(bpm_label)
        knobs_layout.addLayout(bpm_container)
        
        # Top-K Knob
        topk_container = QVBoxLayout()
        self.topk_knob = ModernKnob(1, 100, 40)
        self.topk_knob.valueChanged.connect(self.update_topk)
        topk_container.addWidget(self.topk_knob)
        topk_label = QLabel("TOP-K")
        topk_label.setAlignment(Qt.AlignCenter)
        topk_container.addWidget(topk_label)
        knobs_layout.addLayout(topk_container)
        
        top_control_layout.addLayout(knobs_layout)
        top_control_layout.addSpacing(20)
        
        # Parameters display
        self.params_display = QLabel()
        self.params_display.setFixedSize(180, 80)
        self.params_display.setStyleSheet("""
            QLabel {
                background-color: rgba(28, 28, 28, 220);
                color: #00FF00;
                border: 2px solid #333;
                border-radius: 15px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        self.update_params_display()
        top_control_layout.addWidget(self.params_display)
        top_control_layout.addStretch()
        
        # Main buttons
        self.connect_btn = QPushButton("CONNECT")
        self.connect_btn.setCheckable(True)
        self.connect_btn.clicked.connect(self.connect_to_lyria)
        self.connect_btn.setFixedSize(80, 40)
        
        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.play_music)
        self.play_btn.setEnabled(False)
        self.play_btn.setFixedSize(50, 40)
        
        self.pause_btn = QPushButton("❚❚")
        self.pause_btn.clicked.connect(self.pause_music)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setFixedSize(50, 40)
        
        self.stop_btn = QPushButton("■")
        self.stop_btn.clicked.connect(self.stop_music)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setFixedSize(50, 40)
        
        for btn in [self.play_btn, self.pause_btn, self.stop_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF8C00;
                    color: white;
                    border: none;
                    border-radius: 15px;
                    font-size: 16px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #FF7700;
                }
                QPushButton:pressed {
                    background-color: #E67300;
                }
                QPushButton:disabled {
                    background-color: rgba(204, 204, 204, 150);
                    color: #999;
                }
            """)
        
        top_control_layout.addWidget(self.connect_btn)
        top_control_layout.addWidget(self.play_btn)
        top_control_layout.addWidget(self.pause_btn)
        top_control_layout.addWidget(self.stop_btn)
        control_layout.addLayout(top_control_layout)
        
        # Scale selection
        scale_layout = QHBoxLayout()
        scale_label = QLabel("SCALE:")
        scale_layout.addWidget(scale_label)
        
        self.scale_combo = QComboBox()
        for scale in types.Scale:
            self.scale_combo.addItem(scale.name.replace('_', ' ').title(), scale)
        self.scale_combo.currentIndexChanged.connect(self.update_scale)
        self.scale_combo.setFixedWidth(200)
        scale_layout.addWidget(self.scale_combo)
        scale_layout.addStretch()
        control_layout.addLayout(scale_layout)
        main_layout.addWidget(control_frame)
        
        # Prompt input
        prompt_frame = QFrame()
        prompt_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(222, 222, 222, 150);
                border-radius: 15px;
                padding: 15px;
            }
        """)
        prompt_layout = QVBoxLayout(prompt_frame)
        prompt_input_layout = QHBoxLayout()
        prompt_label = QLabel("PROMPTS:")
        prompt_input_layout.addWidget(prompt_label)
        
        self.weighted_prompts_input = QLineEdit()
        self.weighted_prompts_input.setPlaceholderText("piano:0.8, drums:0.3, bass:0.5")
        self.weighted_prompts_input.returnPressed.connect(self.send_weighted_prompts)
        prompt_input_layout.addWidget(self.weighted_prompts_input)
        
        self.send_btn = QPushButton("SEND")
        self.send_btn.clicked.connect(self.send_weighted_prompts)
        self.send_btn.setEnabled(False)
        self.send_btn.setFixedWidth(80)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF8C00;
                color: white;
                border: none;
                border-radius: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF7700;
            }
            QPushButton:disabled {
                background-color: rgba(204, 204, 204, 150);
                color: #999;
            }
        """)
        prompt_input_layout.addWidget(self.send_btn)
        prompt_layout.addLayout(prompt_input_layout)
        main_layout.addWidget(prompt_frame)
        
        # Sliders section
        sliders_outer_frame = QFrame()
        sliders_outer_frame.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border: 2px solid #000000;
                border-radius: 15px;
            }
        """)
        sliders_outer_frame.setMinimumHeight(420)
        sliders_outer_layout = QVBoxLayout(sliders_outer_frame)
        sliders_outer_layout.setContentsMargins(0, 0, 0, 0)
        sliders_outer_layout.setSpacing(0)
        
        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setMinimumHeight(418)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QScrollArea > QWidget > QWidget {
                background-color: transparent;
            }
            QScrollBar:horizontal {
                border: none;
                background: #F0F0F0;
                height: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #CCC;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #AAA;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                border: none;
                background: none;
            }
        """)
        
        # Container widget - make it fill the entire scroll area
        self.container_widget = QWidget()
        self.container_widget.setStyleSheet("background-color: transparent;")
        self.container_widget.setMinimumHeight(418)  # Fill the height
        self.container_widget.setMinimumWidth(770)   # Fill the width
        
        self.sliders_layout = QHBoxLayout(self.container_widget)
        self.sliders_layout.setSpacing(15)
        self.sliders_layout.setContentsMargins(0, 5, 0, 5)  # Minimal vertical margins
        self.sliders_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        # Create initial semi-opaque area that fills the space
        initial_width = 770  # Fill the entire available width
        self.semi_opaque_area = SemiOpaqueArea(initial_width)
        self.sliders_layout.addWidget(self.semi_opaque_area)
        self.sliders_layout.addStretch()
        
        self.scroll_area.setWidget(self.container_widget)
        sliders_outer_layout.addWidget(self.scroll_area)
        
        main_layout.addWidget(sliders_outer_frame)
        main_layout.addStretch(1)
    
    def update_params_display(self):
        bpm_text = f"BPM: {self.bpm_knob.value if hasattr(self, 'bpm_knob') else 120}"
        topk_text = f"TOP-K: {self.topk_knob.value if hasattr(self, 'topk_knob') else 40}"
        scale_text = f"SCALE: {self.scale_combo.currentText()[:15] if hasattr(self, 'scale_combo') else 'A♭ Major F Minor'}"
        prompts_text = f"PROMPTS: {len(self.weighted_prompts)}"
        display_text = f"{bpm_text}\n{topk_text}\n{scale_text}\n{prompts_text}"
        if hasattr(self, 'params_display'):
            self.params_display.setText(display_text)
    
    def update_status(self, status):
        self.status_display.setText(status.upper())
    
    def handle_error(self, error):
        self.status_display.setText(f"ERROR: {error[:30]}...")
        self.status_display.setStyleSheet("""
            QLabel {
                background-color: rgba(34, 34, 34, 200);
                color: #F00;
                padding: 8px 16px;
                border-radius: 15px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        self.connect_btn.setChecked(False)
        self.enable_controls(False)
    
    def connect_to_lyria(self):
        if self.connect_btn.isChecked():
            if self.audio_thread and self.audio_thread.isRunning():
                return
            self.animate_shadow_color_to_blue()
            self.audio_thread = LyriaAudioThread()
            self.audio_thread.status_update.connect(self.update_status)
            self.audio_thread.error_occurred.connect(self.handle_error)
            self.audio_thread.start()
            self.enable_controls(True)
            self.update_status("CONNECTING...")
        else:
            self.stop_music()
    
    def enable_controls(self, enabled):
        self.play_btn.setEnabled(enabled)
        self.pause_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
    
    def play_music(self):
        if self.audio_thread and self.audio_thread.session:
            if self.audio_thread.loop:
                asyncio.run_coroutine_threadsafe(
                    self.audio_thread.session.play(), 
                    self.audio_thread.loop
                )
            self.update_status("PLAYING")
    
    def pause_music(self):
        if self.audio_thread and self.audio_thread.session:
            if self.audio_thread.loop:
                asyncio.run_coroutine_threadsafe(
                    self.audio_thread.session.pause(), 
                    self.audio_thread.loop
                )
            self.update_status("PAUSED")
    
    def stop_music(self):
        if self.audio_thread:
            self.audio_thread.stop_audio()
            self.audio_thread.wait()
            self.audio_thread = None
        self.connect_btn.setChecked(False)
        self.enable_controls(False)
        self.update_status("STOPPED")
        self.status_display.setStyleSheet("""
            QLabel {
                background-color: rgba(34, 34, 34, 200);
                color: #0F0;
                padding: 8px 16px;
                border-radius: 15px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
    
    def send_weighted_prompts(self):
        prompt_str = self.weighted_prompts_input.text().strip()
        if not prompt_str:
            return
        
        segments = prompt_str.split(',')
        new_prompts = []
        new_sliders = []
        
        for segment_str_raw in segments:
            segment_str = segment_str_raw.strip()
            if not segment_str:
                continue
            
            parts = segment_str.split(':', 1)
            if len(parts) == 2:
                text_p = parts[0].strip()
                weight_s = parts[1].strip()
                
                if text_p:
                    try:
                        weight_f = float(weight_s)
                        new_prompt = types.WeightedPrompt(text=text_p, weight=weight_f)
                        new_prompts.append(new_prompt)
                        self.weighted_prompts.append(new_prompt)
                        
                        # Create slider with proper index
                        slider_index = len(self.weighted_prompts) - 1
                        slider = VerticalSlider(text_p, weight_f)
                        
                        # Store slider reference before connecting signals
                        self.sliders.append(slider)
                        new_sliders.append(slider)
                        
                        # Connect signals with the correct index
                        slider.valueChanged.connect(lambda v, idx=slider_index: self.update_prompt_weight(idx, v))
                        slider.deleteRequested.connect(lambda idx=slider_index: self.delete_slider(idx))
                        
                    except ValueError:
                        continue
        
        if new_sliders:
            # Calculate how much space the sliders will take
            slider_width = 80
            slider_spacing = 15
            total_slider_width = len(self.sliders) * (slider_width + slider_spacing)
            
            # Remove semi-opaque area from layout temporarily
            self.sliders_layout.removeWidget(self.semi_opaque_area)
            
            # Insert new sliders at the beginning
            for i, slider in enumerate(new_sliders):
                self.sliders_layout.insertWidget(i, slider)
            
            # Re-add semi-opaque area at the end
            self.sliders_layout.addWidget(self.semi_opaque_area)
            
            # Force layout update before animation
            self.container_widget.updateGeometry()
            self.sliders_layout.update()
            QTimer.singleShot(0, lambda: self.container_widget.updateGeometry())
            
            # Calculate new semi-opaque area width
            scroll_width = 770
            new_opaque_width = max(100, scroll_width - total_slider_width - 20)
            
            # Animate semi-opaque area shrinking
            self.semi_opaque_area.animate_resize(new_opaque_width)
            
            # Fade in new sliders with staggered timing - delay slightly to let layout settle
            for i, slider in enumerate(new_sliders):
                QTimer.singleShot(150 + i * 100, slider.fade_in)
        
        if new_prompts and self.audio_thread and self.audio_thread.session and self.audio_thread.loop:
            asyncio.run_coroutine_threadsafe(
                self.audio_thread.session.set_weighted_prompts(prompts=self.weighted_prompts),
                self.audio_thread.loop
            )
            self.update_status(f"ADDED {len(new_prompts)} PROMPTS ({len(self.weighted_prompts)} TOTAL)")
            self.weighted_prompts_input.clear()
            self.update_params_display()
    
    def delete_slider(self, index):
        if 0 <= index < len(self.sliders):
            slider = self.sliders[index]
            
            # Disconnect signals before removing
            try:
                slider.valueChanged.disconnect()
                slider.deleteRequested.disconnect()
            except:
                pass
            
            # Remove from layout
            self.sliders_layout.removeWidget(slider)
            slider.setParent(None)
            slider.deleteLater()
            
            # Remove from lists
            self.sliders.pop(index)
            self.weighted_prompts.pop(index)
            
            # Reconnect all remaining slider signals with updated indices
            for i, s in enumerate(self.sliders):
                try:
                    s.valueChanged.disconnect()
                except:
                    pass
                try:
                    s.deleteRequested.disconnect()
                except:
                    pass
                s.valueChanged.connect(lambda v, idx=i: self.update_prompt_weight(idx, v))
                s.deleteRequested.connect(lambda idx=i: self.delete_slider(idx))
            
            # Update semi-opaque area size
            slider_width = 80
            slider_spacing = 15
            total_slider_width = len(self.sliders) * (slider_width + slider_spacing) if self.sliders else 0
            scroll_width = 770
            new_opaque_width = max(100, scroll_width - total_slider_width - 20)
            
            if not self.sliders:
                new_opaque_width = 770
            
            self.semi_opaque_area.animate_resize(new_opaque_width)
            
            if self.audio_thread and self.audio_thread.session and self.audio_thread.loop and self.weighted_prompts:
                asyncio.run_coroutine_threadsafe(
                    self.audio_thread.session.set_weighted_prompts(prompts=self.weighted_prompts),
                    self.audio_thread.loop
                )
            
            self.update_status(f"DELETED SLIDER ({len(self.weighted_prompts)} TOTAL)")
            self.update_params_display()
    
    def update_prompt_weight(self, index, value):
        if 0 <= index < len(self.weighted_prompts):
            self.weighted_prompts[index].weight = value
            if self.audio_thread and self.audio_thread.session and self.audio_thread.loop:
                asyncio.run_coroutine_threadsafe(
                    self.audio_thread.session.set_weighted_prompts(prompts=self.weighted_prompts),
                    self.audio_thread.loop
                )
    
    def update_bpm(self, value):
        if self.audio_thread and self.audio_thread.session and self.audio_thread.loop:
            self.audio_thread.config.bpm = value
            asyncio.run_coroutine_threadsafe(
                self.update_config_and_reset(),
                self.audio_thread.loop
            )
            self.update_status(f"BPM: {value}")
        self.update_params_display()
    
    def update_scale(self, index):
        if self.audio_thread and self.audio_thread.session and self.audio_thread.loop:
            scale = self.scale_combo.itemData(index)
            self.audio_thread.config.scale = scale
            asyncio.run_coroutine_threadsafe(
                self.update_config_and_reset(),
                self.audio_thread.loop
            )
        self.update_params_display()
    
    def update_topk(self, value):
        if self.audio_thread and self.audio_thread.session and self.audio_thread.loop:
            self.audio_thread.config.top_k = value
            asyncio.run_coroutine_threadsafe(
                self.update_config_and_reset(),
                self.audio_thread.loop
            )
        self.update_params_display()
    
    async def update_config_and_reset(self):
        if self.audio_thread and self.audio_thread.session:
            await self.audio_thread.session.set_music_generation_config(config=self.audio_thread.config)
            await self.audio_thread.session.reset_context()
    
    def closeEvent(self, event):
        if self.audio_thread:
            self.audio_thread.stop_audio()
            self.audio_thread.wait()
        event.accept()

    def animate_shadow(self):
        """Animate shadow offset from 0 to 45"""
        if not self.shadow_effect:
            return
            
        # Animate X offset
        x_anim = QPropertyAnimation(self.shadow_effect, b"xOffset")
        x_anim.setDuration(1000)
        x_anim.setStartValue(0.0)
        x_anim.setEndValue(45.0)
        x_anim.setEasingCurve(QEasingCurve.OutCubic)
        
        # Animate Y offset
        y_anim = QPropertyAnimation(self.shadow_effect, b"yOffset")
        y_anim.setDuration(1000)
        y_anim.setStartValue(0.0)
        y_anim.setEndValue(45.0)
        y_anim.setEasingCurve(QEasingCurve.OutCubic)
        
        # Start both animations
        x_anim.start()
        y_anim.start()
        
        # Store references to prevent garbage collection
        self.shadow_x_anim = x_anim
        self.shadow_y_anim = y_anim

    def animate_shadow_color_to_blue(self):
        """Animate shadow color from black to blue when Connect is clicked"""
        if not self.shadow_effect:
            return
        
        # Create a custom animation for color transition
        self.shadow_color_timer = QTimer()
        self.shadow_color_step = 0
        self.shadow_color_steps = 30
        
        def update_color():
            progress = self.shadow_color_step / self.shadow_color_steps
            # Interpolate from black (0,0,0) to blue (100,100,255)
            r = int(0 + (100 - 0) * progress)
            g = int(0 + (100 - 0) * progress)
            b = int(0 + (255 - 0) * progress)
            a = int(160 + (150 - 160) * progress)
            
            self.shadow_effect.setColor(QColor(r, g, b, a))
            
            self.shadow_color_step += 1
            if self.shadow_color_step >= self.shadow_color_steps:
                self.shadow_color_timer.stop()
        
        self.shadow_color_timer.timeout.connect(update_color)
        self.shadow_color_timer.start(30)  # Update every 30ms

# Create the widget instance
lyria_generator = LyriaMusicGenerator()

# Add it to the graphics scene as a proxy widget
lyria_proxy = graphics_scene.addWidget(lyria_generator)

# Optional: Make it movable and selectable
lyria_proxy.setFlags(
    lyria_proxy.flags() | 
    QGraphicsItem.ItemIsMovable | 
    QGraphicsItem.ItemIsSelectable
)

shadow_effect = QGraphicsDropShadowEffect()
shadow_effect.setBlurRadius(20)
shadow_effect.setColor(QColor(0, 0, 0, 160))
shadow_effect.setOffset(0, 0)
lyria_proxy.setGraphicsEffect(shadow_effect)

lyria_generator.shadow_effect = shadow_effect

# Center the widget in the scene
if hasattr(graphics_scene, 'views') and graphics_scene.views():
    view = graphics_scene.views()[0]
    viewport_rect = view.viewport().rect()
    scene_rect = view.mapToScene(viewport_rect).boundingRect()
    
    # Calculate center position
    widget_width = lyria_generator.width()
    widget_height = lyria_generator.height()
    center_x = scene_rect.center().x() - widget_width / 2
    center_y = scene_rect.center().y() - widget_height / 2
    
    lyria_proxy.setPos(center_x, center_y)
else:
    # Fallback: center in scene rect
    scene_rect = graphics_scene.sceneRect()
    center_x = scene_rect.center().x() - lyria_generator.width() / 2
    center_y = scene_rect.center().y() - lyria_generator.height() / 2
    lyria_proxy.setPos(center_x, center_y)

QTimer.singleShot(500, lyria_generator.animate_shadow)