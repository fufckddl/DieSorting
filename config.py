"""
설정 파일: 모델 경로, 기본 파라미터 등
"""

import os
from pathlib import Path

# 프로젝트 루트 디렉터리
PROJECT_ROOT = Path(__file__).parent

# 모델 가중치 경로 (학습 후 설정)
YOLO_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "yolo_detector.pt"
PATTERN_CLASSIFIER_WEIGHTS_PATH = PROJECT_ROOT / "weights" / "pattern_classifier.pt"

# YOLO 모델 설정
YOLO_MODEL_SIZE = "n"  # "n" (nano), "s" (small), "m" (medium), "l" (large), "x" (xlarge)
YOLO_TASK = "detect"  # "detect" 또는 "segment"

# Wafer map 설정
GRID_SIZE = 128  # wafer-map 해상도 (128x128)
GRID_HEIGHT = GRID_SIZE
GRID_WIDTH = GRID_SIZE

# 패턴 분류 클래스
PATTERN_CLASSES = [
    "local",
    "edge-local",
    "donut",
    "edge-ring",
    "scratch",
    "near-full",
    "none"
]

# 시각화 설정
VIS_COLORS = {
    "local": (0, 255, 0),          # 녹색
    "edge-local": (255, 165, 0),   # 주황색
    "donut": (255, 0, 255),        # 자홍색
    "edge-ring": (0, 255, 255),    # 청록색
    "scratch": (255, 0, 0),        # 빨간색
    "near-full": (128, 0, 128),    # 보라색
    "none": (128, 128, 128)        # 회색
}

# YOLO 검출 임계값
CONFIDENCE_THRESHOLD = 0.15  # 낮춰서 더 많은 검출 시도 (0.25 -> 0.15)
IOU_THRESHOLD = 0.45

# 이미지 전처리
IMAGE_MAX_SIZE = 2048  # 최대 이미지 크기 (리사이즈 시 사용)

# 가중치 디렉터리 생성
weights_dir = PROJECT_ROOT / "weights"
weights_dir.mkdir(exist_ok=True)

