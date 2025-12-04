"""
시각화 유틸리티 (Step D)
원본 이미지 + 검출 결과 + 패턴 분류 결과를 시각화
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import config


def create_detection_based_mask(
    detections: List[Dict],
    image_shape: Tuple[int, int],
    pattern_class: str,
    marker_size_factor: float = 50.0
) -> np.ndarray:
    """
    YOLO 검출 결과를 기반으로 실제 불량 위치에 마스크 생성
    검출된 각 불량의 좌표를 기반으로 큰 원형 마커를 그립니다.
    
    Args:
        detections: YOLO 검출 결과 리스트
        image_shape: 이미지 크기 (width, height)
        pattern_class: 패턴 클래스 이름 (색상 결정용)
        marker_size_factor: 마커 크기 조절 인자 (이미지 크기에 비례)
    
    Returns:
        마스크 이미지 (0-255 범위, uint8)
    """
    img_w, img_h = image_shape
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    if not detections:
        return mask
    
    # 이미지 대각선 길이 기반 마커 크기 계산
    diagonal = np.sqrt(img_w**2 + img_h**2)
    base_radius = int(diagonal / marker_size_factor)
    
    for det in detections:
        # 중심 좌표 추출
        if "center" in det:
            cx, cy = det["center"]
        elif "bbox" in det:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
        else:
            continue
        
        cx, cy = int(cx), int(cy)
        
        # 신뢰도에 따라 마커 크기 조절
        conf = det.get("confidence", 0.5)
        radius = int(base_radius * (0.5 + conf))
        
        # 면적에 따라 마커 크기 조절
        if "area" in det:
            area = det["area"]
            area_factor = min(2.0, max(0.5, np.sqrt(area) / 100))
            radius = int(radius * area_factor)
        
        # 패턴별로 마커 모양 조절
        if pattern_class == "scratch":
            # 스크래치: 타원형 또는 선 형태
            if len(detections) > 1:
                # 다른 불량과의 관계 고려
                pass
            cv2.ellipse(mask, (cx, cy), (radius * 2, radius // 2), 0, 0, 360, 255, -1)
        else:
            # 일반: 원형
            cv2.circle(mask, (cx, cy), radius, 255, -1)
    
    return mask


def cluster_detections(
    detections: List[Dict],
    distance_threshold: float = 0.1
) -> List[Tuple[float, float, float]]:
    """
    검출된 불량들을 클러스터링하여 큰 영역 표시
    
    Args:
        detections: 검출 결과 리스트
        distance_threshold: 클러스터링 거리 임계값 (이미지 크기 대비 비율)
    
    Returns:
        클러스터 중심점과 반지름 리스트 [(cx, cy, radius), ...]
    """
    if not detections:
        return []
    
    # 중심점 추출
    centers = []
    areas = []
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
    
    if not centers:
        return []
    
    # 간단한 클러스터링 (최근접 이웃)
    clusters = []
    used = [False] * len(centers)
    
    for i, (cx, cy) in enumerate(centers):
        if used[i]:
            continue
        
        # 클러스터 초기화
        cluster_points = [(cx, cy)]
        cluster_areas = [areas[i]]
        used[i] = True
        
        # 거리 임계값 내의 점들 찾기
        for j, (cx2, cy2) in enumerate(centers):
            if used[j] or i == j:
                continue
            
            # 거리 계산 (정규화)
            dist = np.sqrt((cx - cx2)**2 + (cy - cy2)**2)
            max_dim = max(np.sqrt(sum(areas)) * 2, 100)  # 대략적인 이미지 크기 추정
            normalized_dist = dist / max_dim
            
            if normalized_dist < distance_threshold:
                cluster_points.append((cx2, cy2))
                cluster_areas.append(areas[j])
                used[j] = True
        
        # 클러스터 중심과 반지름 계산
        if cluster_points:
            avg_x = np.mean([p[0] for p in cluster_points])
            avg_y = np.mean([p[1] for p in cluster_points])
            max_dist = max([np.sqrt((p[0] - avg_x)**2 + (p[1] - avg_y)**2) 
                           for p in cluster_points] + [20])
            avg_area = np.mean(cluster_areas)
            radius = max(max_dist * 1.5, np.sqrt(avg_area) / 2)
            
            clusters.append((avg_x, avg_y, radius))
    
    return clusters


def visualize_detections(
    image: np.ndarray,
    detections: List[Dict],
    pattern_class: str,
    pattern_confidence: float = 1.0,
    show_bbox: bool = False,
    show_detection_mask: bool = True,
    alpha: float = 0.5,
    use_clustering: bool = True
) -> np.ndarray:
    """
    검출 결과와 패턴 분류 결과를 원본 이미지에 오버레이
    실제 YOLO 검출 좌표를 기반으로 시각화합니다.
    
    Args:
        image: 원본 이미지 (BGR, uint8)
        detections: YOLO 검출 결과
        pattern_class: 분류된 패턴 클래스
        pattern_confidence: 패턴 분류 신뢰도
        show_bbox: bounding box 표시 여부 (기본: False)
        show_detection_mask: 검출 기반 마스크 표시 여부
        alpha: 마스크 투명도 (0.0 ~ 1.0)
        use_clustering: 클러스터링 사용 여부
    
    Returns:
        시각화된 이미지 (BGR, uint8)
    """
    vis_image = image.copy()
    
    h, w = vis_image.shape[:2]
    
    # 패턴 색상 가져오기
    pattern_color = config.VIS_COLORS.get(pattern_class, (255, 255, 255))
    
    # 1. 검출 결과 기반 마스크 생성 및 오버레이
    if show_detection_mask and pattern_class != "none" and detections:
        if use_clustering and len(detections) > 1:
            # 클러스터링 사용
            clusters = cluster_detections(detections, distance_threshold=0.15)
            mask = np.zeros((h, w), dtype=np.uint8)
            
            for cx, cy, radius in clusters:
                cx, cy, radius = int(cx), int(cy), int(radius)
                # 큰 원형 마커 그리기
                cv2.circle(mask, (cx, cy), radius, 255, -1)
            
            # 컬러 마스크 생성
            color_mask = np.zeros_like(vis_image)
            color_mask[mask > 0] = pattern_color
            
            # 알파 블렌딩
            vis_image = cv2.addWeighted(vis_image, 1.0 - alpha, color_mask, alpha, 0)
        else:
            # 개별 검출 위치에 마커 표시
            mask = create_detection_based_mask(
                detections, 
                (w, h), 
                pattern_class,
                marker_size_factor=30.0  # 마커 크기 조절 (작을수록 큰 마커)
            )
            
            if mask.max() > 0:
                # 컬러 마스크 생성
                color_mask = np.zeros_like(vis_image)
                color_mask[mask > 0] = pattern_color
                
                # 알파 블렌딩
                vis_image = cv2.addWeighted(vis_image, 1.0 - alpha, color_mask, alpha, 0)
    
    # 2. 검출 bbox 그리기 (선택적)
    if show_bbox and detections:
        for det in detections:
            if "bbox" in det:
                x1, y1, x2, y2 = [int(x) for x in det["bbox"]]
                conf = det.get("confidence", 0.0)
                
                # bbox 색상 (신뢰도에 따라)
                if conf > 0.7:
                    bbox_color = (0, 255, 0)  # 녹색 (높은 신뢰도)
                elif conf > 0.5:
                    bbox_color = (0, 165, 255)  # 주황색 (중간 신뢰도)
                else:
                    bbox_color = (0, 0, 255)  # 빨간색 (낮은 신뢰도)
                
                # bbox 그리기 (두께 2로 표시)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), bbox_color, 2)
                
                # 신뢰도 텍스트 표시
                label = f"{conf:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # 텍스트 배경
                cv2.rectangle(
                    vis_image,
                    (x1, y1 - text_h - 4),
                    (x1 + text_w, y1),
                    bbox_color,
                    -1
                )
                # 텍스트
                cv2.putText(
                    vis_image,
                    label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
    
    # 3. 패턴 정보 텍스트 추가
    text = f"Pattern: {pattern_class} ({pattern_confidence:.2f})"
    cv2.putText(
        vis_image, 
        text, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.0, 
        (0, 255, 255), 
        2
    )
    
    return vis_image


def create_visualization(
    image: np.ndarray,
    detections: List[Dict],
    pattern_class: str,
    pattern_confidence: float = 1.0,
    show_bbox: bool = False,
    show_detection_mask: bool = True,
    alpha: float = 0.5,
    use_clustering: bool = True,
    **kwargs
) -> np.ndarray:
    """
    최종 시각화 이미지 생성 (편의 함수)
    YOLO 검출 좌표를 기반으로 실제 불량 위치에 큰 마커를 표시합니다.
    
    Args:
        image: 원본 이미지 (BGR)
        detections: 검출 결과 (YOLO bbox/center 좌표 포함)
        pattern_class: 패턴 클래스
        pattern_confidence: 패턴 신뢰도
        show_bbox: bounding box 표시 여부
        show_detection_mask: 검출 기반 마스크 표시 여부
        alpha: 마스크 투명도
        use_clustering: 클러스터링 사용 여부
        **kwargs: 추가 인자
    
    Returns:
        시각화된 이미지 (BGR)
    """
    return visualize_detections(
        image, 
        detections, 
        pattern_class, 
        pattern_confidence,
        show_bbox=show_bbox,
        show_detection_mask=show_detection_mask,
        alpha=alpha,
        use_clustering=use_clustering,
        **kwargs
    )

