"""
Wafer map 생성 유틸리티 (Step B)
YOLO 검출 결과를 칩 좌표계/그리드에 매핑하여 wafer-map 형태의 2D 배열 생성
"""

import numpy as np
from typing import List, Dict, Tuple
import config


def detections_to_wafer_map(
    detections: List[Dict],
    image_shape: Tuple[int, int],
    grid_size: Tuple[int, int] = (config.GRID_HEIGHT, config.GRID_WIDTH)
) -> np.ndarray:
    """
    YOLO 검출 결과를 wafer-map 형태의 2D 그리드로 변환
    
    이 함수는 불량 위치를 정규화된 그리드 좌표계에 매핑하여,
    패턴 분류 모델이 입력으로 사용할 수 있는 분포 맵을 생성합니다.
    
    Args:
        detections: YOLO 검출 결과 리스트. 각 요소는:
            {
                "bbox": [x1, y1, x2, y2] 또는
                "center": (cx, cy),
                "area": float,
                ...
            }
        image_shape: 원본 이미지 크기 (width, height)
        grid_size: 생성할 그리드 크기 (height, width)
    
    Returns:
        2D numpy 배열 (grid_height, grid_width)
        - 각 셀의 값은 해당 영역의 불량 개수 또는 밀도
        - 값이 클수록 해당 영역에 불량이 많음
    """
    grid_h, grid_w = grid_size
    img_w, img_h = image_shape
    
    # 초기화: 불량 개수 맵
    wafer_map = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    if not detections:
        return wafer_map
    
    # 그리드 셀 크기
    cell_w = img_w / grid_w
    cell_h = img_h / grid_h
    
    for det in detections:
        # 중심 좌표 사용
        if "center" in det:
            cx, cy = det["center"]
        elif "bbox" in det:
            # bbox에서 중심 계산
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
        else:
            continue
        
        # 그리드 좌표로 변환
        grid_x = int(cx / cell_w)
        grid_y = int(cy / cell_h)
        
        # 경계 체크
        grid_x = np.clip(grid_x, 0, grid_w - 1)
        grid_y = np.clip(grid_y, 0, grid_h - 1)
        
        # 불량 영역 가중치 (면적 기반)
        if "area" in det:
            # 정규화된 면적 (이미지 크기에 대한 비율)
            normalized_area = det["area"] / (img_w * img_h)
            weight = min(1.0, normalized_area * 100)  # 스케일링
        else:
            weight = 1.0
        
        # 신뢰도 가중치
        if "confidence" in det:
            weight *= det["confidence"]
        
        # 그리드 셀에 누적
        wafer_map[grid_y, grid_x] += weight
    
    # 정규화 (선택적)
    if wafer_map.max() > 0:
        wafer_map = wafer_map / wafer_map.max()
    
    return wafer_map


def create_binary_wafer_map(
    detections: List[Dict],
    image_shape: Tuple[int, int],
    grid_size: Tuple[int, int] = (config.GRID_HEIGHT, config.GRID_WIDTH),
    threshold: float = 0.0
) -> np.ndarray:
    """
    이진 wafer-map 생성 (불량 존재 여부만 표시)
    
    Args:
        detections: YOLO 검출 결과
        image_shape: 이미지 크기 (width, height)
        grid_size: 그리드 크기 (height, width)
        threshold: 불량 존재 임계값
    
    Returns:
        2D 이진 배열 (0 또는 1)
    """
    wafer_map = detections_to_wafer_map(detections, image_shape, grid_size)
    binary_map = (wafer_map > threshold).astype(np.float32)
    return binary_map


def create_density_wafer_map(
    detections: List[Dict],
    image_shape: Tuple[int, int],
    grid_size: Tuple[int, int] = (config.GRID_HEIGHT, config.GRID_WIDTH),
    kernel_size: int = 3
) -> np.ndarray:
    """
    밀도 기반 wafer-map 생성 (주변 영역도 고려)
    
    Args:
        detections: YOLO 검출 결과
        image_shape: 이미지 크기
        grid_size: 그리드 크기
        kernel_size: 가우시안 커널 크기
    
    Returns:
        2D 밀도 맵
    """
    wafer_map = detections_to_wafer_map(detections, image_shape, grid_size)
    
    # 가우시안 필터로 스무딩 (선택적)
    if kernel_size > 0:
        from scipy import ndimage
        try:
            wafer_map = ndimage.gaussian_filter(wafer_map, sigma=kernel_size / 3)
        except ImportError:
            # scipy가 없으면 스킵
            pass
    
    return wafer_map


