"""
학습 데이터 생성 유틸리티
YOLO 검출 결과로부터 wafer-map을 생성하여 패턴 분류 학습 데이터 준비
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

from models.yolo_detector import load_yolo_detector
from utils.wafer_map import detections_to_wafer_map
import config


def generate_wafer_maps_from_images(
    image_dir: str,
    output_dir: str,
    yolo_weights: str = None,
    conf_threshold: float = 0.25
):
    """
    이미지 디렉터리에서 YOLO로 검출 후 wafer-map 생성
    
    Args:
        image_dir: 원본 이미지 디렉터리
        output_dir: wafer-map 저장 디렉터리
        yolo_weights: YOLO 가중치 경로 (None이면 사전학습 모델)
        conf_threshold: YOLO 신뢰도 임계값
    """
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # YOLO 검출기 로드
    print("YOLO 검출기 로딩 중...")
    detector = load_yolo_detector(yolo_weights)
    
    # 이미지 파일 수집
    image_files = list(image_path.glob("*.jpg")) + \
                  list(image_path.glob("*.png")) + \
                  list(image_path.glob("*.jpeg"))
    
    print(f"총 {len(image_files)}개 이미지 처리 시작...")
    
    metadata = []
    
    for img_file in tqdm(image_files):
        try:
            # 이미지 로드
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"경고: 이미지를 로드할 수 없습니다: {img_file}")
                continue
            
            h, w = img.shape[:2]
            
            # YOLO 검출
            detections = detector.detect(img, conf_threshold=conf_threshold)
            
            # Wafer-map 생성
            wafer_map = detections_to_wafer_map(
                detections,
                (w, h),
                grid_size=(config.GRID_HEIGHT, config.GRID_WIDTH)
            )
            
            # 이미지로 저장 (0-255 범위로 변환)
            wafer_map_img = (wafer_map * 255).astype(np.uint8)
            
            # PIL Image로 변환하여 저장
            output_file = output_path / f"{img_file.stem}_wafermap.png"
            Image.fromarray(wafer_map_img).save(output_file)
            
            # 메타데이터 저장
            metadata.append({
                "original_image": str(img_file),
                "wafer_map": str(output_file),
                "detections_count": len(detections),
                "image_size": (w, h)
            })
            
        except Exception as e:
            print(f"오류 발생 ({img_file}): {e}")
            continue
    
    # 메타데이터 저장
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ 완료!")
    print(f"생성된 wafer-map: {len(metadata)}개")
    print(f"저장 위치: {output_path}")
    print(f"\n다음 단계:")
    print(f"1. 생성된 wafer-map 이미지들을 패턴별로 디렉터리에 분류")
    print(f"2. train_pattern_classifier.py로 학습")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 검출 결과로부터 wafer-map 생성"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="원본 이미지 디렉터리"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="wafer-map 저장 디렉터리"
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="YOLO 가중치 경로 (기본: 사전학습 모델)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="YOLO 신뢰도 임계값"
    )
    
    args = parser.parse_args()
    
    generate_wafer_maps_from_images(
        args.image_dir,
        args.output_dir,
        args.yolo_weights,
        args.conf_threshold
    )


if __name__ == "__main__":
    main()

