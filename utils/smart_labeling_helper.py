"""
스마트 라벨링 도우미
현재 YOLO로 자동 검출 후 검증/수정하는 반자동 라벨링 도구
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json

from models.yolo_detector import load_yolo_detector
import config


def auto_detect_and_save_labels(
    image_dir: str,
    output_labels_dir: str = None,
    conf_threshold: float = 0.25,
    min_confidence: float = 0.3
):
    """
    현재 YOLO 모델로 자동 검출하여 라벨 파일 생성
    사용자는 생성된 라벨을 검증하고 수정만 하면 됨
    
    Args:
        image_dir: 이미지 디렉터리
        output_labels_dir: 라벨 파일 저장 디렉터리 (None이면 이미지와 같은 위치)
        conf_threshold: YOLO 신뢰도 임계값
        min_confidence: 저장할 최소 신뢰도 (너무 낮은 신뢰도는 제외)
    """
    image_path = Path(image_dir)
    
    if output_labels_dir is None:
        labels_path = image_path
    else:
        labels_path = Path(output_labels_dir)
        labels_path.mkdir(parents=True, exist_ok=True)
    
    # YOLO 검출기 로드
    print("YOLO 검출기 로딩 중...")
    detector = load_yolo_detector()
    
    # 이미지 파일 찾기
    image_files = list(image_path.glob("*.jpg")) + \
                  list(image_path.glob("*.png")) + \
                  list(image_path.glob("*.jpeg"))
    
    print(f"\n총 {len(image_files)}개 이미지 처리 시작...")
    print(f"신뢰도 {min_confidence} 이상만 저장됩니다.\n")
    
    stats = {
        "total": len(image_files),
        "processed": 0,
        "with_detections": 0,
        "no_detections": 0,
        "low_confidence": 0
    }
    
    for img_file in tqdm(image_files):
        try:
            # 이미지 로드
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # YOLO 검출
            detections = detector.detect(img, conf_threshold=conf_threshold)
            
            # 신뢰도 필터링
            valid_detections = [
                det for det in detections 
                if det.get("confidence", 0) >= min_confidence
            ]
            
            if not valid_detections:
                stats["no_detections"] += 1
                # 빈 라벨 파일 생성 (선택적)
                # labels_path / f"{img_file.stem}.txt" 으로 빈 파일 생성 가능
                continue
            
            # YOLO 형식 라벨 생성
            label_lines = []
            for det in valid_detections:
                if "bbox" in det:
                    x1, y1, x2, y2 = det["bbox"]
                elif "center" in det:
                    # 중심에서 bbox 생성 (면적 추정)
                    cx, cy = det["center"]
                    area = det.get("area", 100)
                    size = np.sqrt(area)
                    x1 = max(0, cx - size/2)
                    y1 = max(0, cy - size/2)
                    x2 = min(w, cx + size/2)
                    y2 = min(h, cy + size/2)
                else:
                    continue
                
                # 정규화된 좌표로 변환
                center_x = ((x1 + x2) / 2.0) / w
                center_y = ((y1 + y2) / 2.0) / h
                bbox_width = (x2 - x1) / w
                bbox_height = (y2 - y1) / h
                
                # 클래스 ID: 0 (defect)
                label_lines.append(f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
            
            # 라벨 파일 저장
            if label_lines:
                label_file = labels_path / f"{img_file.stem}.txt"
                with open(label_file, 'w') as f:
                    f.writelines(label_lines)
                
                stats["with_detections"] += 1
            else:
                stats["low_confidence"] += 1
            
            stats["processed"] += 1
        
        except Exception as e:
            print(f"\n오류 발생 ({img_file}): {e}")
            continue
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("자동 검출 완료 통계")
    print("=" * 60)
    print(f"전체 이미지: {stats['total']}개")
    print(f"처리 완료: {stats['processed']}개")
    print(f"검출 성공: {stats['with_detections']}개")
    print(f"검출 실패: {stats['no_detections']}개")
    print(f"신뢰도 낮음: {stats['low_confidence']}개")
    print("=" * 60)
    
    print("\n다음 단계:")
    print("1. 생성된 라벨 파일 확인")
    print("2. LabelImg로 검증 및 수정")
    print("   - 검출되지 않은 불량 추가")
    print("   - 잘못된 박스 수정/삭제")
    print("3. 수정이 필요한 이미지만 편집하면 효율적!")
    
    return stats


def process_all_patterns(
    dataset_dir: str = "dataset",
    conf_threshold: float = 0.25,
    min_confidence: float = 0.3
):
    """모든 패턴 폴더에서 자동 검출 실행"""
    dataset_path = Path(dataset_dir)
    
    print("=" * 60)
    print("모든 패턴 폴더 자동 검출 시작")
    print("=" * 60)
    
    total_stats = {
        "total": 0,
        "processed": 0,
        "with_detections": 0,
        "no_detections": 0,
        "low_confidence": 0
    }
    
    # 패턴별 디렉터리 처리
    pattern_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    for pattern_dir in sorted(pattern_dirs):
        pattern_name = pattern_dir.name
        print(f"\n[{pattern_name}] 처리 중...")
        
        stats = auto_detect_and_save_labels(
            str(pattern_dir),
            conf_threshold=conf_threshold,
            min_confidence=min_confidence
        )
        
        # 통계 누적
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)
    
    # 전체 통계
    print("\n" + "=" * 60)
    print("전체 통계")
    print("=" * 60)
    print(f"전체 이미지: {total_stats['total']}개")
    print(f"검출 성공: {total_stats['with_detections']}개 ({total_stats['with_detections']/total_stats['total']*100:.1f}%)")
    print(f"검출 실패: {total_stats['no_detections']}개 ({total_stats['no_detections']/total_stats['total']*100:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="YOLO로 자동 검출하여 라벨 파일 생성 (반자동 라벨링)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="데이터셋 디렉터리 (패턴별 폴더 포함)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="특정 패턴 폴더만 처리 (예: Donut)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="YOLO 신뢰도 임계값"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="저장할 최소 신뢰도"
    )
    
    args = parser.parse_args()
    
    if args.pattern:
        # 특정 패턴만 처리
        pattern_dir = Path(args.dataset_dir) / args.pattern
        if pattern_dir.exists():
            auto_detect_and_save_labels(
                str(pattern_dir),
                conf_threshold=args.conf_threshold,
                min_confidence=args.min_confidence
            )
        else:
            print(f"패턴 디렉터리를 찾을 수 없습니다: {pattern_dir}")
    else:
        # 모든 패턴 처리
        process_all_patterns(
            args.dataset_dir,
            args.conf_threshold,
            args.min_confidence
        )


if __name__ == "__main__":
    main()


