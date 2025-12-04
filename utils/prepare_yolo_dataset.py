"""
YOLO 학습용 데이터셋 준비 스크립트
패턴별로 분류된 이미지에서 YOLO 학습용 데이터셋 생성
"""

import argparse
import shutil
from pathlib import Path
import cv2
import yaml
from tqdm import tqdm

# 패턴별 디렉터리명 매핑 (대소문자, 공백 처리)
PATTERN_MAPPING = {
    "Local": "local",
    "Center": "local",  # Center를 local로 매핑
    "Edge Local": "edge-local",
    "EdgeLocal": "edge-local",
    "Donut": "donut",
    "Edge Ring": "edge-ring",
    "EdgeRing": "edge-ring",
    "Scratch": "scratch",
    "near full": "near-full",
    "nearfull": "near-full",
    "none": "none"
}


def create_yolo_dataset_from_patterns(
    source_dir: str,
    output_dir: str,
    class_name: str = "defect",
    train_split: float = 0.8
):
    """
    패턴별로 분류된 이미지에서 YOLO 학습용 데이터셋 생성
    
    주의: 이 스크립트는 각 이미지 전체를 하나의 불량으로 간주합니다.
    더 정확한 학습을 위해서는 실제 bbox 라벨링이 필요합니다.
    
    Args:
        source_dir: 패턴별로 분류된 이미지 디렉터리
        output_dir: YOLO 데이터셋 출력 디렉터리
        class_name: 클래스 이름 (기본: "defect")
        train_split: 학습 데이터 비율
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉터리 구조 생성
    train_images = output_path / "train" / "images"
    train_labels = output_path / "train" / "labels"
    val_images = output_path / "val" / "images"
    val_labels = output_path / "val" / "labels"
    
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"소스 디렉터리: {source_path}")
    print(f"출력 디렉터리: {output_path}")
    print(f"\n⚠️  주의: 각 이미지 전체를 하나의 불량으로 간주합니다.")
    print(f"더 정확한 학습을 위해서는 실제 bbox 라벨링이 필요합니다.\n")
    
    # 패턴별 디렉터리 처리
    all_images = []
    
    for pattern_dir in source_path.iterdir():
        if not pattern_dir.is_dir():
            continue
        
        pattern_name = pattern_dir.name
        
        # 이미지 파일 수집
        images = list(pattern_dir.glob("*.jpg")) + \
                 list(pattern_dir.glob("*.png")) + \
                 list(pattern_dir.glob("*.jpeg"))
        
        for img_path in images:
            all_images.append((img_path, pattern_name))
    
    print(f"총 {len(all_images)}개 이미지 발견")
    
    # Train/Val 분할
    import random
    random.seed(42)
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * train_split)
    train_images_list = all_images[:split_idx]
    val_images_list = all_images[split_idx:]
    
    print(f"학습 데이터: {len(train_images_list)}개")
    print(f"검증 데이터: {len(val_images_list)}개\n")
    
    # 이미지 처리 함수
    def process_images(images_list, img_dir, label_dir, split_name):
        for img_path, pattern_name in tqdm(images_list, desc=f"{split_name} 처리 중"):
            # 이미지 복사
            img_filename = img_path.name
            dest_img_path = img_dir / img_filename
            shutil.copy(img_path, dest_img_path)
            
            # 이미지 크기 확인
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"경고: 이미지를 로드할 수 없습니다: {img_path}")
                continue
            
            h, w = img.shape[:2]
            
            # YOLO 형식 라벨 생성 (전체 이미지를 하나의 불량으로 간주)
            # 중심: (0.5, 0.5), 크기: (0.9, 0.9) - 약간의 마진
            label_content = f"0 0.5 0.5 0.9 0.9\n"  # class_id=0 (defect), 정규화된 좌표
            
            # 라벨 파일 저장
            label_filename = img_path.stem + ".txt"
            label_path = label_dir / label_filename
            
            with open(label_path, 'w') as f:
                f.write(label_content)
    
    # 처리
    process_images(train_images_list, train_images, train_labels, "학습")
    process_images(val_images_list, val_images, val_labels, "검증")
    
    # dataset.yaml 파일 생성
    dataset_yaml = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: class_name
        },
        'nc': 1  # 클래스 개수
    }
    
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ 데이터셋 준비 완료!")
    print(f"출력 위치: {output_path}")
    print(f"YAML 파일: {yaml_path}")
    print(f"\n다음 단계:")
    print(f"python train_yolo.py --data {yaml_path} --epochs 100")
    print(f"\n⚠️  참고: 현재는 이미지 전체를 불량으로 간주합니다.")
    print(f"더 정확한 학습을 위해서는 각 불량의 실제 bbox 좌표가 필요합니다.")


def main():
    parser = argparse.ArgumentParser(
        description="패턴별 이미지에서 YOLO 학습용 데이터셋 생성"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="dataset",
        help="패턴별로 분류된 이미지 디렉터리 (기본: dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="yolo_dataset",
        help="YOLO 데이터셋 출력 디렉터리 (기본: yolo_dataset)"
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="defect",
        help="클래스 이름 (기본: defect)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="학습 데이터 비율 (기본: 0.8)"
    )
    
    args = parser.parse_args()
    
    create_yolo_dataset_from_patterns(
        args.source_dir,
        args.output_dir,
        args.class_name,
        args.train_split
    )


if __name__ == "__main__":
    main()


