"""
PyQt 메인 윈도우
반도체 불량 패턴 분류 GUI 애플리케이션
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont

from inference.pipeline import run_full_pipeline


class AnalysisThread(QThread):
    """분석 작업을 별도 스레드에서 실행"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path
    
    def run(self):
        try:
            result = run_full_pipeline(self.image_path)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """메인 윈도우 클래스"""
    
    def __init__(self):
        super().__init__()
        self.current_image_path: Optional[Path] = None
        self.current_result: Optional[dict] = None
        self.zoom_factor = 1.0
        
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("반도체 불량 패턴 분류 시스템")
        self.setGeometry(100, 100, 1400, 900)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃 (2x2 그리드)
        main_layout = QGridLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. 좌상단: 불량률 패널
        defect_rate_panel = self.create_defect_rate_panel()
        main_layout.addWidget(defect_rate_panel, 0, 0)
        
        # 2. 우상단: 이미지 뷰어 패널
        image_viewer_panel = self.create_image_viewer_panel()
        main_layout.addWidget(image_viewer_panel, 0, 1)
        
        # 3. 좌하단: 상세 불량 결과 패널
        detail_panel = self.create_detail_panel()
        main_layout.addWidget(detail_panel, 1, 0)
        
        # 4. 우하단: Settings 패널
        settings_panel = self.create_settings_panel()
        main_layout.addWidget(settings_panel, 1, 1)
        
        # 메뉴바
        menubar = self.menuBar()
        file_menu = menubar.addMenu('파일')
        
        open_action = file_menu.addAction('이미지 열기')
        open_action.triggered.connect(self.open_image)
        
        exit_action = file_menu.addAction('종료')
        exit_action.triggered.connect(self.close)
        
        # 스타일 설정 (다크 테마)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                padding: 5px;
                color: #ffffff;
                min-width: 200px;
            }
            QPushButton {
                background-color: #4a4a4a;
                border: 1px solid #666666;
                padding: 8px;
                color: #ffffff;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QTableWidget {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                color: #ffffff;
                gridline-color: #555555;
            }
            QHeaderView::section {
                background-color: #4a4a4a;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #555555;
            }
        """)
    
    def create_defect_rate_panel(self) -> QWidget:
        """불량률 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 제목
        title = QLabel("불량률")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # 파일명
        layout.addWidget(QLabel("파일명:"))
        self.filename_edit = QLineEdit()
        self.filename_edit.setReadOnly(True)
        self.filename_edit.setMinimumWidth(200)  # 최소 너비 설정
        layout.addWidget(self.filename_edit)
        
        # 분석 시간
        layout.addWidget(QLabel("분석 시간:"))
        self.analysis_time_edit = QLineEdit()
        self.analysis_time_edit.setReadOnly(True)
        self.analysis_time_edit.setMinimumWidth(200)
        layout.addWidget(self.analysis_time_edit)
        
        # 이미지 크기
        layout.addWidget(QLabel("이미지 크기:"))
        self.image_size_edit = QLineEdit()
        self.image_size_edit.setReadOnly(True)
        self.image_size_edit.setMinimumWidth(200)
        layout.addWidget(self.image_size_edit)
        
        # 패턴 분류 결과
        layout.addWidget(QLabel("분류된 패턴:"))
        self.pattern_edit = QLineEdit()
        self.pattern_edit.setReadOnly(True)
        self.pattern_edit.setMinimumWidth(200)
        layout.addWidget(self.pattern_edit)
        
        # 패턴 신뢰도
        layout.addWidget(QLabel("신뢰도:"))
        self.confidence_edit = QLineEdit()
        self.confidence_edit.setReadOnly(True)
        self.confidence_edit.setMinimumWidth(200)
        layout.addWidget(self.confidence_edit)
        
        # 분석 시작 버튼
        self.analyze_button = QPushButton("분석 시작")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setMinimumHeight(40)
        layout.addWidget(self.analyze_button)
        
        layout.addStretch()
        
        return panel
    
    def create_image_viewer_panel(self) -> QWidget:
        """이미지 뷰어 패널 생성"""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setSpacing(5)
        
        # 스크롤 영역
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        
        # 이미지 라벨
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("이미지를 선택하세요")
        self.image_label.setMinimumSize(600, 400)
        scroll_area.setWidget(self.image_label)
        
        layout.addWidget(scroll_area)
        
        # 우측 버튼 패널
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)
        
        # Zoom In 버튼
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(40, 40)
        zoom_in_btn.clicked.connect(self.zoom_in)
        button_layout.addWidget(zoom_in_btn)
        
        # Zoom Out 버튼
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setFixedSize(40, 40)
        zoom_out_btn.clicked.connect(self.zoom_out)
        button_layout.addWidget(zoom_out_btn)
        
        # 리셋 버튼
        reset_btn = QPushButton("Reset")
        reset_btn.setFixedSize(50, 40)  # 너비 증가하여 텍스트가 잘리지 않도록
        reset_btn.clicked.connect(self.reset_zoom)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return panel
    
    def create_detail_panel(self) -> QWidget:
        """상세 불량 결과 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 제목
        title = QLabel("상세 불량 결과")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # 패턴 분류 정보 섹션
        pattern_frame = QWidget()
        pattern_layout = QVBoxLayout(pattern_frame)
        pattern_layout.setSpacing(5)
        pattern_layout.setContentsMargins(5, 5, 5, 5)
        
        pattern_label = QLabel("패턴 분류:")
        pattern_label_font = QFont()
        pattern_label_font.setBold(True)
        pattern_label.setFont(pattern_label_font)
        pattern_layout.addWidget(pattern_label)
        
        # 패턴 정보를 표시할 텍스트 영역
        self.pattern_info_text = QLabel("분석 전")
        self.pattern_info_text.setWordWrap(True)
        self.pattern_info_text.setMinimumHeight(150)  # 최소 높이 설정
        self.pattern_info_text.setStyleSheet("""
            QLabel {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                padding: 8px;
                border-radius: 3px;
            }
        """)
        pattern_layout.addWidget(self.pattern_info_text)
        
        layout.addWidget(pattern_frame)
        
        # 구분선
        separator = QLabel("────────────────────────")
        separator.setStyleSheet("color: #555555;")
        layout.addWidget(separator)
        
        # 개별 불량 검출 결과
        detection_label = QLabel("개별 불량 검출 결과:")
        detection_label_font = QFont()
        detection_label_font.setBold(True)
        detection_label.setFont(detection_label_font)
        layout.addWidget(detection_label)
        
        # 테이블
        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(3)
        self.detail_table.setHorizontalHeaderLabels(["Defect Type", "Position (X,Y)", "Confidence"])
        # 컬럼 너비 조정 (텍스트가 잘리지 않도록)
        self.detail_table.setColumnWidth(0, 180)  # Defect Type
        self.detail_table.setColumnWidth(1, 180)  # Position
        self.detail_table.setColumnWidth(2, 120)  # Confidence
        # 테이블이 컬럼 너비를 자동 조정하도록 설정
        self.detail_table.horizontalHeader().setStretchLastSection(True)
        # 텍스트가 잘리지 않도록 설정
        self.detail_table.setTextElideMode(Qt.ElideNone)
        layout.addWidget(self.detail_table)
        
        return panel
    
    def create_settings_panel(self) -> QWidget:
        """Settings 패널 생성 (stub)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 제목
        title = QLabel("Settings")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # 향후 설정 추가 예정
        label = QLabel("향후 설정 추가 예정")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        layout.addStretch()
        
        return panel
    
    def open_image(self):
        """이미지 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "이미지 선택",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.current_image_path = Path(file_path)
            self.filename_edit.setText(self.current_image_path.name)
            
            # 이미지 크기 표시
            img = cv2.imread(str(self.current_image_path))
            if img is not None:
                h, w = img.shape[:2]
                self.image_size_edit.setText(f"{w} × {h}")
                
                # 원본 이미지 미리보기
                self.display_image(img)
            
            # 초기화
            self.analysis_time_edit.clear()
            self.pattern_edit.clear()
            self.confidence_edit.clear()
            self.confidence_edit.setStyleSheet("")  # 스타일 초기화
            self.pattern_info_text.setText("분석 전")
            self.detail_table.setRowCount(0)
            self.current_result = None
            self.zoom_factor = 1.0
    
    def display_image(self, image: np.ndarray, reset_zoom: bool = False):
        """이미지를 QLabel에 표시 (작은 이미지는 640x640으로 확대)"""
        if reset_zoom:
            self.zoom_factor = 1.0
        
        # BGR → RGB 변환
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        h, w = rgb_image.shape[:2]
        
        # 이미지를 150x150으로 리사이즈 (pipeline에서 이미 150x150으로 생성되지만, 원본 이미지 미리보기용으로 유지)
        if w != 150 or h != 150:
            # 정확히 150x150으로 리사이즈
            rgb_image = cv2.resize(rgb_image, (150, 150), interpolation=cv2.INTER_CUBIC)
            h, w = 150, 150
        
        # 확대/축소 적용
        if self.zoom_factor != 1.0:
            new_w = int(w * self.zoom_factor)
            new_h = int(h * self.zoom_factor)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = new_h, new_w
        
        # QImage로 변환
        bytes_per_line = 3 * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # QLabel에 표시
        self.image_label.setPixmap(pixmap)
    
    def zoom_in(self):
        """확대"""
        if self.current_result is not None:
            self.zoom_factor = min(self.zoom_factor * 1.2, 5.0)
            self.display_image(self.current_result["vis_image"])
        elif self.current_image_path is not None:
            img = cv2.imread(str(self.current_image_path))
            if img is not None:
                self.zoom_factor = min(self.zoom_factor * 1.2, 5.0)
                self.display_image(img)
    
    def zoom_out(self):
        """축소"""
        if self.current_result is not None:
            self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)
            self.display_image(self.current_result["vis_image"])
        elif self.current_image_path is not None:
            img = cv2.imread(str(self.current_image_path))
            if img is not None:
                self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)
                self.display_image(img)
    
    def reset_zoom(self):
        """줌 리셋"""
        if self.current_result is not None:
            self.display_image(self.current_result["vis_image"], reset_zoom=True)
        elif self.current_image_path is not None:
            img = cv2.imread(str(self.current_image_path))
            if img is not None:
                self.display_image(img, reset_zoom=True)
    
    def start_analysis(self):
        """분석 시작"""
        if self.current_image_path is None:
            QMessageBox.warning(self, "경고", "먼저 이미지를 선택하세요.")
            return
        
        # 버튼 비활성화
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("분석 중...")
        
        # 분석 스레드 시작
        self.analysis_thread = AnalysisThread(str(self.current_image_path))
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.start()
    
    def on_analysis_finished(self, result: dict):
        """분석 완료 처리"""
        self.current_result = result
        
        # 분석 시간 표시
        analysis_time = result.get("analysis_time", 0.0)
        self.analysis_time_edit.setText(f"{analysis_time:.2f}초")
        
        # 이미지 크기 업데이트 (이미 설정되어 있을 수 있음)
        img_size = result.get("image_size", (0, 0))
        if img_size[0] > 0 and img_size[1] > 0:
            self.image_size_edit.setText(f"{img_size[0]} × {img_size[1]}")
        
        # 패턴 분류 결과 표시
        pattern_class = result.get("class_label", "unknown")
        pattern_confidence = result.get("confidence", 0.0)
        self.pattern_edit.setText(pattern_class)
        self.confidence_edit.setText(f"{pattern_confidence:.2%}")
        
        # 신뢰도가 낮으면 경고 색상 적용
        if pattern_confidence < 0.3:
            self.confidence_edit.setStyleSheet("background-color: #5a2a2a; color: #ff6b6b;")
        elif pattern_confidence < 0.5:
            self.confidence_edit.setStyleSheet("background-color: #5a4a2a; color: #ffb86b;")
        else:
            self.confidence_edit.setStyleSheet("")  # 기본 스타일
        
        # 상세 불량 결과 패널에 패턴 분류 정보 표시
        pattern_probs = result.get("pattern_probabilities", {})
        if pattern_probs:
            # 패턴 정보 텍스트 생성
            info_text = f"<b>분류된 패턴:</b> {pattern_class}<br>"
            info_text += f"<b>신뢰도:</b> {pattern_confidence:.2%}"
            
            # 신뢰도에 따른 색상
            if pattern_confidence < 0.3:
                confidence_color = "#ff6b6b"  # 빨간색
                confidence_level = " (매우 낮음)"
            elif pattern_confidence < 0.5:
                confidence_color = "#ffb86b"  # 주황색
                confidence_level = " (낮음)"
            elif pattern_confidence < 0.7:
                confidence_color = "#ffd86b"  # 노란색
                confidence_level = " (보통)"
            else:
                confidence_color = "#6bff6b"  # 녹색
                confidence_level = " (높음)"
            
            info_text += f' <span style="color: {confidence_color};">{confidence_level}</span><br><br>'
            info_text += "<b>전체 확률 분포:</b><br>"
            
            # 확률 내림차순 정렬
            sorted_probs = sorted(pattern_probs.items(), key=lambda x: x[1], reverse=True)
            # 모든 패턴 표시
            for i, (pattern, prob) in enumerate(sorted_probs):
                if pattern == pattern_class:
                    info_text += f"• <b>{pattern}</b>: {prob:.2%} ← <span style='color: {confidence_color};'>선택됨</span><br>"
                else:
                    info_text += f"• {pattern}: {prob:.2%}<br>"
            
            self.pattern_info_text.setText(info_text)
            
            # 터미널 출력 (디버깅용)
            print("\n[패턴 분류 확률 분포]")
            for pattern, prob in sorted_probs:
                marker = "← 선택됨" if pattern == pattern_class else ""
                print(f"  {pattern:15s}: {prob:6.2%} {marker}")
            print()
        else:
            # 확률 정보가 없는 경우
            info_text = f"<b>분류된 패턴:</b> {pattern_class}<br>"
            info_text += f"<b>신뢰도:</b> {pattern_confidence:.2%}"
            self.pattern_info_text.setText(info_text)
        
        # 시각화된 이미지 표시
        vis_image = result.get("vis_image")
        if vis_image is not None:
            self.zoom_factor = 1.0
            self.display_image(vis_image, reset_zoom=True)
        
        # 상세 불량 결과 테이블 업데이트
        detections = result.get("detections", [])
        self.detail_table.setRowCount(len(detections))
        
        for row, det in enumerate(detections):
            # Defect Type
            defect_type = det.get("class_name", "unknown")
            item_type = QTableWidgetItem(str(defect_type))
            item_type.setTextAlignment(Qt.AlignCenter)
            self.detail_table.setItem(row, 0, item_type)
            
            # Position
            center = det.get("center", (0, 0))
            pos_str = f"({int(center[0])}, {int(center[1])})"
            item_pos = QTableWidgetItem(pos_str)
            item_pos.setTextAlignment(Qt.AlignCenter)
            self.detail_table.setItem(row, 1, item_pos)
            
            # Confidence
            conf = det.get("confidence", 0.0)
            conf_str = f"{conf:.3f}"
            item_conf = QTableWidgetItem(conf_str)
            item_conf.setTextAlignment(Qt.AlignCenter)
            self.detail_table.setItem(row, 2, item_conf)
        
        # 테이블 크기 조정 (텍스트가 잘리지 않도록)
        self.detail_table.resizeColumnsToContents()
        
        # 버튼 활성화
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("분석 시작")
    
    def on_analysis_error(self, error_msg: str):
        """분석 오류 처리"""
        QMessageBox.critical(self, "오류", f"분석 중 오류가 발생했습니다:\n{error_msg}")
        
        # 버튼 활성화
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("분석 시작")

