"""
규칙 기반 패턴 분류기
YOLO 검출 결과의 공간적 분포를 기반으로 패턴을 분류
학습된 모델이 없을 때 사용
"""
import numpy as np
from typing import List, Dict, Tuple
import config


def classify_pattern_by_rules(
    detections: List[Dict],
    image_shape: Tuple[int, int],
    return_probabilities: bool = False
) -> Dict:
    """
    검출 결과의 공간적 분포를 기반으로 패턴 분류
    
    Args:
        detections: YOLO 검출 결과
        image_shape: 이미지 크기 (width, height)
        return_probabilities: 확률 반환 여부
    
    Returns:
        패턴 분류 결과
    """
    if not detections:
        result = {
            "class": "local",  # 기본값
            "confidence": 0.5,
            "probabilities": {cls: 1.0 / len(config.PATTERN_CLASSES) for cls in config.PATTERN_CLASSES}
        }
        return result if return_probabilities else result["class"]
    
    img_w, img_h = image_shape
    center_x, center_y = img_w / 2, img_h / 2
    
    # 검출 위치 분석
    centers = []
    areas = []
    confidences = []
    
    for det in detections:
        if "center" in det:
            cx, cy = det["center"]
        elif "bbox" in det:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
        else:
            continue
        
        centers.append((cx, cy))
        areas.append(det.get("area", 100))
        confidences.append(det.get("confidence", 0.5))
    
    if not centers:
        result = {
            "class": "local",
            "confidence": 0.5,
            "probabilities": {cls: 1.0 / len(config.PATTERN_CLASSES) for cls in config.PATTERN_CLASSES}
        }
        return result if return_probabilities else result["class"]
    
    # 중심점에서의 거리 계산
    distances_from_center = [
        np.sqrt((cx - center_x)**2 + (cy - center_y)**2) 
        for cx, cy in centers
    ]
    max_distance = np.sqrt(center_x**2 + center_y**2)
    normalized_distances = [d / max_distance for d in distances_from_center]
    
    # 가장자리 근처인지 확인 (이미지 가장자리 20% 영역)
    edge_threshold = 0.2
    edge_detections = sum(1 for d in normalized_distances if d > (1 - edge_threshold))
    edge_ratio = edge_detections / len(centers)
    
    # 중심 근처인지 확인 (이미지 중심 30% 영역)
    center_threshold = 0.3
    center_detections = sum(1 for d in normalized_distances if d < center_threshold)
    center_ratio = center_detections / len(centers)
    
    # 평균 거리
    avg_distance = np.mean(normalized_distances)
    
    # 검출 개수
    num_detections = len(centers)
    
    # 패턴 분류 규칙
    pattern_scores = {}
    
    # Local: 중심에 집중된 검출
    if center_ratio > 0.6 and avg_distance < 0.3:
        pattern_scores["local"] = 0.8
    else:
        pattern_scores["local"] = 0.2
    
    # Edge-local: 가장자리에 집중된 검출
    if edge_ratio > 0.6 and avg_distance > 0.7:
        pattern_scores["edge-local"] = 0.8
    else:
        pattern_scores["edge-local"] = 0.2
    
    # Donut: 중심이 비어있고 주변에 검출
    if center_ratio < 0.2 and 0.3 < avg_distance < 0.7:
        pattern_scores["donut"] = 0.8
    else:
        pattern_scores["donut"] = 0.2
    
    # Edge-ring: 가장자리 전체에 고르게 분포
    if edge_ratio > 0.7 and 0.6 < avg_distance < 0.9:
        pattern_scores["edge-ring"] = 0.8
    else:
        pattern_scores["edge-ring"] = 0.2
    
    # Scratch: 선형적으로 분포된 검출 (간단한 휴리스틱)
    if num_detections >= 3:
        # 검출들이 일직선상에 있는지 확인 (간단한 방법)
        centers_array = np.array(centers)
        # 주성분 분석으로 선형성 확인
        if len(centers_array) > 2:
            centered = centers_array - centers_array.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvals = np.linalg.eigvals(cov)
            linearity = eigenvals[0] / (eigenvals.sum() + 1e-6) if eigenvals.sum() > 0 else 0
            if linearity > 0.7:  # 선형적 분포
                pattern_scores["scratch"] = 0.7
            else:
                pattern_scores["scratch"] = 0.2
        else:
            pattern_scores["scratch"] = 0.2
    else:
        pattern_scores["scratch"] = 0.2
    
    # 확률 정규화
    total_score = sum(pattern_scores.values())
    probabilities = {cls: pattern_scores.get(cls, 0.1) / total_score for cls in config.PATTERN_CLASSES}
    
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

