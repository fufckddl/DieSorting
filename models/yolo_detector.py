"""
YOLOv8 기반 불량 검출 모델 (Step A)
칩/웨이퍼 이미지에서 개별 불량의 위치를 검출
"""

import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("경고: ultralytics가 설치되지 않았습니다. pip install ultralytics로 설치하세요.")

import config


class YOLODetector:
    """YOLOv8 불량 검출기"""
    
    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        """
        Args:
            weights_path: YOLO 모델 가중치 경로. None이면 사전학습 모델 사용
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics 패키지가 필요합니다.")
        
        if weights_path is None:
            weights_path = config.YOLO_WEIGHTS_PATH
        
        weights_path = Path(weights_path)
        
        # 가중치 파일이 없으면 사전학습 모델로 초기화
        if not weights_path.exists():
            print(f"경고: {weights_path}를 찾을 수 없습니다. 사전학습 모델(yolov8{config.YOLO_MODEL_SIZE}.pt)을 사용합니다.")
            model_name = f"yolov8{config.YOLO_MODEL_SIZE}.pt"
            self.model = YOLO(model_name)
        else:
            self.model = YOLO(str(weights_path))
    
    def detect(
        self, 
        image: Union[np.ndarray, str, Path],
        conf_threshold: float = config.CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.IOU_THRESHOLD
    ) -> List[Dict]:
        """
        이미지에서 불량을 검출
        
        Args:
            image: 이미지 경로 또는 numpy 배열 (BGR 또는 RGB)
            conf_threshold: 신뢰도 임계값
            iou_threshold: IoU 임계값 (NMS)
        
        Returns:
            검출 결과 리스트. 각 요소는:
            {
                "bbox": [x1, y1, x2, y2],  # bounding box 좌표
                "confidence": float,        # 신뢰도 점수
                "class_id": int,            # 클래스 ID
                "class_name": str,          # 클래스 이름 (있는 경우)
                "center": (cx, cy),         # 중심 좌표
                "area": float               # 영역 크기
            }
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image}")
        else:
            img = image.copy()
            # RGB인 경우 BGR로 변환
            if len(img.shape) == 3 and img.shape[2] == 3:
                # OpenCV는 BGR을 기대하므로, RGB라면 변환 필요
                # (이미 BGR이라면 변환 불필요)
                pass
        
        # YOLO 추론
        results = self.model(
            img,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        detections = []
        result = results[0]  # 첫 번째 결과
        
        # 검출 결과 파싱
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(confidences[i]),
                    "class_id": int(class_ids[i]),
                    "class_name": self.model.names[class_ids[i]] if hasattr(self.model, 'names') else "defect",
                    "center": (float(cx), float(cy)),
                    "area": float(area)
                })
        
        return detections
    
    def get_image_shape(self, image: Union[np.ndarray, str, Path]) -> tuple:
        """이미지 크기 반환 (width, height)"""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image}")
        else:
            img = image
        
        h, w = img.shape[:2]
        return (w, h)


def load_yolo_detector(weights_path: Optional[Union[str, Path]] = None) -> YOLODetector:
    """
    YOLO 검출기 로드
    
    Args:
        weights_path: 모델 가중치 경로
    
    Returns:
        YOLODetector 인스턴스
    """
    return YOLODetector(weights_path)


def infer_defects(
    model: YOLODetector,
    image: Union[np.ndarray, str, Path],
    conf_threshold: float = config.CONFIDENCE_THRESHOLD
) -> List[Dict]:
    """
    단일 이미지에서 불량 검출 (편의 함수)
    
    Args:
        model: YOLODetector 인스턴스
        image: 이미지 경로 또는 numpy 배열
        conf_threshold: 신뢰도 임계값
    
    Returns:
        검출 결과 리스트
    """
    return model.detect(image, conf_threshold=conf_threshold)


