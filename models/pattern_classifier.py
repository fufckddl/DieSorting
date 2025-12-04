"""
패턴 분류 모델 (Step C)
불량 분포 맵을 입력으로 받아 패턴 클래스로 분류
"""

import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

import config


class SimplePatternClassifier(nn.Module):
    """
    간단한 CNN 기반 패턴 분류 모델
    wafer-map 이미지를 입력으로 받아 패턴 클래스를 분류
    """
    
    def __init__(self, num_classes: int = len(config.PATTERN_CLASSES)):
        super().__init__()
        
        # 간단한 CNN 구조
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class PatternClassifier:
    """패턴 분류기 래퍼 클래스"""
    
    def __init__(
        self, 
        weights_path: Optional[Union[str, Path]] = None,
        use_yolo_cls: bool = False
    ):
        """
        Args:
            weights_path: 모델 가중치 경로
            use_yolo_cls: True면 YOLOv8-cls 사용, False면 PyTorch CNN 사용
        """
        self.use_yolo_cls = use_yolo_cls
        self.num_classes = len(config.PATTERN_CLASSES)
        
        if use_yolo_cls and ULTRALYTICS_AVAILABLE:
            # YOLOv8-cls 모델 사용
            if weights_path is None:
                weights_path = config.PATTERN_CLASSIFIER_WEIGHTS_PATH
            
            weights_path = Path(weights_path)
            
            if weights_path.exists():
                self.model = YOLO(str(weights_path))
            else:
                print(f"경고: {weights_path}를 찾을 수 없습니다. 사전학습 모델(yolov8{config.YOLO_MODEL_SIZE}-cls.pt)을 사용합니다.")
                model_name = f"yolov8{config.YOLO_MODEL_SIZE}-cls.pt"
                self.model = YOLO(model_name)
        else:
            # PyTorch CNN 모델 사용
            self.model = SimplePatternClassifier(self.num_classes)
            
            if weights_path is None:
                weights_path = config.PATTERN_CLASSIFIER_WEIGHTS_PATH
            
            weights_path = Path(weights_path)
            
            if weights_path.exists():
                self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
    
    def predict(
        self, 
        wafer_map: np.ndarray,
        return_probabilities: bool = False
    ) -> Union[str, Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        wafer-map을 기반으로 패턴 분류
        
        Args:
            wafer_map: 2D numpy 배열 (grid_height, grid_width)
            return_probabilities: True면 모든 클래스의 확률도 반환
        
        Returns:
            패턴 클래스 이름 또는 {
                "class": str,
                "confidence": float,
                "probabilities": {class_name: prob, ...}
            }
        """
        if self.use_yolo_cls and ULTRALYTICS_AVAILABLE:
            # YOLOv8-cls 사용
            # wafer_map을 이미지로 변환 (0-255 범위)
            img = (wafer_map * 255).astype(np.uint8)
            img = np.stack([img] * 3, axis=-1)  # RGB로 변환 (YOLO는 3채널 기대)
            
            results = self.model(img, verbose=False)
            result = results[0]
            
            if hasattr(result, 'probs'):
                probs = result.probs.data.cpu().numpy()
                class_id = int(probs.argmax())
                confidence = float(probs[class_id])
                class_name = config.PATTERN_CLASSES[class_id] if class_id < len(config.PATTERN_CLASSES) else "unknown"
            else:
                # 폴백
                class_name = "none"
                confidence = 0.0
                probs = np.zeros(self.num_classes)
        else:
            # PyTorch CNN 사용
            # 전처리
            img_tensor = torch.from_numpy(wafer_map).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            class_name = config.PATTERN_CLASSES[class_id] if class_id < len(config.PATTERN_CLASSES) else "unknown"
        
        if return_probabilities:
            prob_dict = {
                config.PATTERN_CLASSES[i]: float(probs[i]) 
                for i in range(min(len(config.PATTERN_CLASSES), len(probs)))
            }
            return {
                "class": class_name,
                "confidence": confidence,
                "probabilities": prob_dict
            }
        else:
            return class_name


def load_pattern_classifier(
    weights_path: Optional[Union[str, Path]] = None,
    use_yolo_cls: bool = False
) -> PatternClassifier:
    """
    패턴 분류기 로드
    
    Args:
        weights_path: 모델 가중치 경로
        use_yolo_cls: YOLOv8-cls 사용 여부
    
    Returns:
        PatternClassifier 인스턴스
    """
    return PatternClassifier(weights_path, use_yolo_cls)


def predict_pattern(
    model: PatternClassifier,
    wafer_map: np.ndarray,
    return_probabilities: bool = False
) -> Union[str, Dict]:
    """
    패턴 예측 (편의 함수)
    
    Args:
        model: PatternClassifier 인스턴스
        wafer_map: 불량 분포 맵
        return_probabilities: 확률 반환 여부
    
    Returns:
        패턴 클래스 또는 결과 딕셔너리
    """
    return model.predict(wafer_map, return_probabilities)


