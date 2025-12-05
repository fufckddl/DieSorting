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


class ImprovedPatternClassifier(nn.Module):
    """
    개선된 패턴 분류 모델 (학습된 모델과 일치)
    - 더 깊은 네트워크
    - Batch Normalization
    - Dropout
    """
    def __init__(self, num_classes: int = len(config.PATTERN_CLASSES)):
        super().__init__()
        
        # 더 깊은 CNN 구조
        self.features = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # 두 번째 블록
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # 세 번째 블록
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # 네 번째 블록
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


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
            if weights_path is None:
                weights_path = config.PATTERN_CLASSIFIER_WEIGHTS_PATH
            
            weights_path = Path(weights_path)
            
            if weights_path.exists():
                # 먼저 가중치를 로드하여 모델 타입 감지
                checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
                
                # ImprovedPatternClassifier인지 확인 (classifier.10.weight가 있으면 Improved)
                is_improved = isinstance(checkpoint, dict) and 'classifier.10.weight' in checkpoint
                
                if is_improved:
                    # ImprovedPatternClassifier 사용
                    self.model = ImprovedPatternClassifier(self.num_classes)
                    print("✅ ImprovedPatternClassifier 모델 사용 (학습된 모델과 일치)")
                else:
                    # SimplePatternClassifier 사용 (이전 모델 호환성)
                    self.model = SimplePatternClassifier(self.num_classes)
                    print("✅ SimplePatternClassifier 모델 사용 (이전 모델 호환)")
                
                # 모델 가중치 로드
                try:
                    self.model.load_state_dict(checkpoint, strict=False)
                    self.model.eval()
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.model.to(self.device)
                    self.use_trained_model = True
                except Exception as e:
                    print(f"⚠️  모델 가중치 로드 중 오류: {e}")
                    print("⚠️  규칙 기반 패턴 분류를 사용합니다.")
                    self.use_trained_model = False
                    self.model = None
            else:
                # 가중치가 없으면 SimplePatternClassifier 사용
                self.model = SimplePatternClassifier(self.num_classes)
                print(f"경고: {weights_path}를 찾을 수 없습니다. 규칙 기반 패턴 분류를 사용합니다.")
                self.use_trained_model = False
                self.model = None
    
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
            # 학습된 모델이 없으면 규칙 기반 분류 사용
            if not self.use_trained_model:
                from utils.pattern_classifier_rule_based import classify_pattern_by_rules
                # 규칙 기반 분류는 detections가 필요하므로, 여기서는 wafer_map만으로는 불가
                # 대신 간단한 휴리스틱 사용
                return self._predict_by_heuristic(wafer_map, return_probabilities)
            
            # PyTorch CNN 사용
            # 전처리
            img_tensor = torch.from_numpy(wafer_map).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # 모든 패턴 클래스 사용 (near-full, none 포함)
            valid_patterns = config.PATTERN_CLASSES
            valid_probs = probs[:len(valid_patterns)] if len(probs) >= len(valid_patterns) else probs
            
            # local 클래스 편향 해소: local 확률에 temperature 적용
            local_idx = valid_patterns.index('local') if 'local' in valid_patterns else -1
            if local_idx >= 0 and local_idx < len(valid_probs):
                # local 확률을 낮춤 (temperature = 1.5)
                valid_probs[local_idx] = valid_probs[local_idx] ** (1.0 / 1.5)
                # 다른 클래스 확률을 상대적으로 높임
                other_indices = [i for i in range(len(valid_probs)) if i != local_idx]
                for idx in other_indices:
                    valid_probs[idx] = valid_probs[idx] ** (1.0 / 0.9)  # 다른 클래스 확률 증가
            
            # 확률 정규화
            total_prob = valid_probs.sum()
            if total_prob > 0:
                valid_probs = valid_probs / total_prob
            
            class_id = int(np.argmax(valid_probs))
            confidence = float(valid_probs[class_id])
            class_name = valid_patterns[class_id] if class_id < len(valid_patterns) else "local"
        
        if return_probabilities:
            # 모든 패턴 클래스 사용 (near-full, none 포함)
            valid_patterns = config.PATTERN_CLASSES
            prob_dict = {}
            
            # 모든 클래스의 확률 포함
            for i, pattern in enumerate(valid_patterns):
                if i < len(probs):
                    prob_dict[pattern] = float(probs[i])
            
            # local 클래스 편향 해소: local 확률에 temperature 적용
            if 'local' in prob_dict:
                # local 확률을 낮춤 (temperature = 1.5)
                prob_dict['local'] = prob_dict['local'] ** (1.0 / 1.5)
                # 다른 클래스 확률을 상대적으로 높임
                for pattern in prob_dict:
                    if pattern != 'local':
                        prob_dict[pattern] = prob_dict[pattern] ** (1.0 / 0.9)  # 다른 클래스 확률 증가
            
            # 확률 정규화
            total_prob = sum(prob_dict.values())
            if total_prob > 0:
                prob_dict = {k: v / total_prob for k, v in prob_dict.items()}
            
            # 가장 높은 확률의 패턴 선택
            if prob_dict:
                best_pattern = max(prob_dict.items(), key=lambda x: x[1])
                class_name = best_pattern[0]
                confidence = best_pattern[1]
            
            return {
                "class": class_name,
                "confidence": confidence,
                "probabilities": prob_dict
            }
        else:
            # class_name이 valid_patterns에 없으면 첫 번째로 변경
            valid_patterns = config.PATTERN_CLASSES
            if class_name not in valid_patterns:
                # probs에서 가장 높은 valid 패턴 선택
                valid_probs = [probs[i] if i < len(probs) else 0.0 for i in range(len(valid_patterns))]
                best_idx = int(np.argmax(valid_probs))
                class_name = valid_patterns[best_idx]
            return class_name
    
    def _predict_by_heuristic(self, wafer_map: np.ndarray, return_probabilities: bool = False):
        """
        Wafer-map의 휴리스틱 기반 패턴 분류 (학습된 모델이 없을 때 사용)
        """
        h, w = wafer_map.shape
        center_y, center_x = h // 2, w // 2
        
        # 중심 영역 (30% 반경)
        center_radius = min(h, w) * 0.3
        edge_threshold = min(h, w) * 0.2
        
        # 중심 영역 밀도
        center_mask = np.zeros_like(wafer_map)
        y_coords, x_coords = np.ogrid[:h, :w]
        center_mask = ((y_coords - center_y)**2 + (x_coords - center_x)**2) < center_radius**2
        center_density = wafer_map[center_mask].sum() if center_mask.sum() > 0 else 0
        
        # 가장자리 영역 밀도
        edge_mask = (
            (y_coords < edge_threshold) | (y_coords > h - edge_threshold) |
            (x_coords < edge_threshold) | (x_coords > w - edge_threshold)
        )
        edge_density = wafer_map[edge_mask].sum() if edge_mask.sum() > 0 else 0
        
        # 전체 밀도
        total_density = wafer_map.sum()
        
        # 패턴 점수 계산
        scores = {}
        
        # Local: 중심에 집중
        if center_density > total_density * 0.6:
            scores["local"] = 0.8
        else:
            scores["local"] = 0.2
        
        # Edge-local: 가장자리에 집중
        if edge_density > total_density * 0.6:
            scores["edge-local"] = 0.8
        else:
            scores["edge-local"] = 0.2
        
        # Donut: 중심이 비어있고 주변에 있음
        if center_density < total_density * 0.2 and 0.3 < (center_density + edge_density) / total_density < 0.7:
            scores["donut"] = 0.8
        else:
            scores["donut"] = 0.2
        
        # Edge-ring: 가장자리 전체에 고르게
        if edge_density > total_density * 0.7:
            scores["edge-ring"] = 0.8
        else:
            scores["edge-ring"] = 0.2
        
        # Scratch: 선형 분포 (간단히 높은 밀도 영역이 선형적으로 분포)
        if total_density > 0:
            # 밀도가 높은 영역의 분산으로 선형성 판단
            high_density_mask = wafer_map > wafer_map.mean() * 1.5
            if high_density_mask.sum() > 0:
                y_coords_high, x_coords_high = np.where(high_density_mask)
                if len(y_coords_high) > 2:
                    # 주성분 분석으로 선형성 확인
                    coords = np.column_stack([y_coords_high, x_coords_high])
                    centered = coords - coords.mean(axis=0)
                    cov = np.cov(centered.T)
                    eigenvals = np.linalg.eigvals(cov)
                    if eigenvals.sum() > 0:
                        linearity = eigenvals.max() / eigenvals.sum()
                        if linearity > 0.7:
                            scores["scratch"] = 0.7
                        else:
                            scores["scratch"] = 0.2
                    else:
                        scores["scratch"] = 0.2
                else:
                    scores["scratch"] = 0.2
            else:
                scores["scratch"] = 0.2
        else:
            scores["scratch"] = 0.2
        
        # 확률 정규화
        total_score = sum(scores.values())
        probabilities = {cls: scores.get(cls, 0.1) / total_score for cls in config.PATTERN_CLASSES}
        
        # 가장 높은 확률의 패턴 선택
        best_pattern = max(probabilities.items(), key=lambda x: x[1])
        pattern_class = best_pattern[0]
        confidence = best_pattern[1]
        
        if return_probabilities:
            return {
                "class": pattern_class,
                "confidence": confidence,
                "probabilities": probabilities
            }
        else:
            return pattern_class


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


