"""
전체 파이프라인 추론 모듈
Step A (YOLO 검출) → Step B (Wafer map 생성) → Step C (패턴 분류) → Step D (시각화)
"""

import time
from typing import Dict, Optional, Union
from pathlib import Path
import numpy as np
import cv2

from models.yolo_detector import YOLODetector, load_yolo_detector
from models.pattern_classifier import PatternClassifier, load_pattern_classifier
from utils.wafer_map import detections_to_wafer_map
from utils.visualization import create_visualization

import config


class DefectAnalysisPipeline:
    """불량 분석 파이프라인"""
    
    def __init__(
        self,
        yolo_weights: Optional[Union[str, Path]] = None,
        classifier_weights: Optional[Union[str, Path]] = None,
        use_yolo_cls: bool = False
    ):
        """
        Args:
            yolo_weights: YOLO 모델 가중치 경로
            classifier_weights: 패턴 분류기 가중치 경로
            use_yolo_cls: YOLOv8-cls 사용 여부
        """
        print("파이프라인 초기화 중...")
        
        # Step A: YOLO 검출기 로드
        self.yolo_detector = load_yolo_detector(yolo_weights)
        print("YOLO 검출기 로드 완료")
        
        # Step C: 패턴 분류기 로드
        self.pattern_classifier = load_pattern_classifier(classifier_weights, use_yolo_cls)
        print("패턴 분류기 로드 완료")
    
    def run(
        self,
        image_path: Union[str, Path],
        conf_threshold: float = config.CONFIDENCE_THRESHOLD
    ) -> Dict:
        """
        전체 파이프라인 실행
        
        Args:
            image_path: 분석할 이미지 경로
            conf_threshold: YOLO 신뢰도 임계값
        
        Returns:
            결과 딕셔너리:
            {
                "class_label": str,              # 패턴 클래스 이름
                "confidence": float,             # 패턴 분류 신뢰도
                "orig_image": np.ndarray,        # 원본 이미지 (BGR)
                "vis_image": np.ndarray,         # 시각화된 이미지 (BGR)
                "detections": List[Dict],        # YOLO 검출 결과
                "analysis_time": float,          # 분석 소요 시간 (초)
                "image_size": (width, height),   # 이미지 크기
                "wafer_map": np.ndarray,         # 생성된 wafer-map
                "pattern_probabilities": Dict    # 각 패턴의 확률 (선택적)
            }
        """
        start_time = time.time()
        image_path = Path(image_path)
        
        # 이미지 로드
        orig_image = cv2.imread(str(image_path))
        if orig_image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        img_h, img_w = orig_image.shape[:2]
        image_size = (img_w, img_h)
        
        # Step A: YOLO 불량 검출
        detections = self.yolo_detector.detect(orig_image, conf_threshold=conf_threshold)
        print(f"검출된 불량 개수: {len(detections)}")
        
        # Step B: Wafer map 생성
        wafer_map = detections_to_wafer_map(
            detections,
            image_size,
            grid_size=(config.GRID_HEIGHT, config.GRID_WIDTH)
        )
        print(f"Wafer map 생성 완료: {wafer_map.shape}")
        
        # Step C: 패턴 분류
        pattern_result = self.pattern_classifier.predict(
            wafer_map,
            return_probabilities=True
        )
        
        if isinstance(pattern_result, dict):
            pattern_class = pattern_result["class"]
            pattern_confidence = pattern_result["confidence"]
            pattern_probs = pattern_result.get("probabilities", {})
        else:
            pattern_class = pattern_result
            pattern_confidence = 1.0
            pattern_probs = {}
        
        print(f"분류된 패턴: {pattern_class} (신뢰도: {pattern_confidence:.2f})")
        
        # Step D: 시각화 (실제 검출 좌표 기반)
        # YOLO 학습 후에는 show_bbox=True로 설정하여 정확한 박스 표시
        vis_image = create_visualization(
            orig_image,
            detections,
            pattern_class,
            pattern_confidence,
            show_bbox=True,  # 학습된 YOLO 모델 사용 시 bbox 표시
            show_detection_mask=True,  # 검출 기반 마스크 사용
            alpha=0.4,  # 투명도 조절 (bbox도 보이도록 약간 낮춤)
            use_clustering=False  # bbox 표시 시 클러스터링 비활성화
        )
        
        analysis_time = time.time() - start_time
        
        # 결과 반환
        return {
            "class_label": pattern_class,
            "confidence": pattern_confidence,
            "orig_image": orig_image,
            "vis_image": vis_image,
            "detections": detections,
            "analysis_time": analysis_time,
            "image_size": image_size,
            "wafer_map": wafer_map,
            "pattern_probabilities": pattern_probs
        }


# 전역 파이프라인 인스턴스 (선택적, 캐싱용)
_pipeline_instance: Optional[DefectAnalysisPipeline] = None


def run_full_pipeline(
    image_path: Union[str, Path],
    yolo_weights: Optional[Union[str, Path]] = None,
    classifier_weights: Optional[Union[str, Path]] = None,
    use_yolo_cls: bool = False,
    conf_threshold: float = config.CONFIDENCE_THRESHOLD
) -> Dict:
    """
    전체 파이프라인 실행 (편의 함수)
    
    주석: YOLO만으로는 bbox 기반 검출만 가능하므로, donut/edge-ring/near-full 같은
    분포형 패턴을 직접 설명하기 어렵습니다. 따라서 YOLO(불량 위치 검출) + wafer-map
    (분포 맵) + 별도 분류 모델(패턴 인식) 구조를 사용하여, 공간적 분포를 고려한
    패턴 분류가 가능하도록 설계했습니다.
    
    Args:
        image_path: 분석할 이미지 경로
        yolo_weights: YOLO 모델 가중치 (None이면 config에서 자동 로드)
        classifier_weights: 패턴 분류기 가중치 (None이면 config에서 자동 로드)
        use_yolo_cls: YOLOv8-cls 사용 여부
        conf_threshold: YOLO 신뢰도 임계값
    
    Returns:
        결과 딕셔너리 (DefectAnalysisPipeline.run()과 동일)
    """
    global _pipeline_instance
    
    # 파이프라인 인스턴스 생성 (재사용)
    if _pipeline_instance is None:
        _pipeline_instance = DefectAnalysisPipeline(
            yolo_weights=yolo_weights,
            classifier_weights=classifier_weights,
            use_yolo_cls=use_yolo_cls
        )
    
    return _pipeline_instance.run(image_path, conf_threshold=conf_threshold)

