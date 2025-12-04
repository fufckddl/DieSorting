"""
Roboflow에서 다운로드한 라벨 파일을 현재 프로젝트에 통합하는 유틸리티
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def import_roboflow_labels(
    roboflow_dir: str,
    target_dataset_dir: str = "dataset",
    copy_images: bool = False
):
    """
    Roboflow에서 다운로드한 라벨 파일을 dataset/ 폴더에 통합
    
    Args:
        roboflow_dir: Roboflow에서 다운로드한 데이터셋 디렉터리
        target_dataset_dir: 대상 dataset 디렉터리
        copy_images: 이미지도 복사할지 여부 (False면 라벨만 복사)
    """
    roboflow_path = Path(roboflow_dir)
    target_path = Path(target_dataset_dir)
    
    if not roboflow_path.exists():
        raise FileNotFoundError(f"Roboflow 디렉터리를 찾을 수 없습니다: {roboflow_dir}")
    
    print("Roboflow 라벨 파일 통합 시작...")
    print(f"소스: {roboflow_path}")
    print(f"대상: {target_path}")
    print()
    
    # Train/Val 폴더 확인
    train_labels = roboflow_path / "train" / "labels"
    val_labels = roboflow_path / "val" / "labels"
    train_images = roboflow_path / "train" / "images"
    val_images = roboflow_path / "val" / "images"
    
    stats = {
        "labels_copied": 0,
        "images_copied": 0,
        "labels_not_found": 0,
        "images_not_found": 0
    }
    
    # 라벨 파일 복사 (train + val)
    for split_dir, labels_dir, images_dir in [
        ("train", train_labels, train_images),
        ("val", val_labels, val_images)
    ]:
        if not labels_dir.exists():
            print(f"⚠️  {split_dir}/labels 폴더를 찾을 수 없습니다: {labels_dir}")
            continue
        
        print(f"[{split_dir}] 처리 중...")
        
        # 각 라벨 파일 처리
        label_files = list(labels_dir.glob("*.txt"))
        
        for label_file in tqdm(label_files, desc=f"{split_dir} 라벨 복사"):
            # 이미지 파일명 추출
            image_name = label_file.stem  # .txt 확장자 제거
            
            # 이미지 파일 찾기 (여러 확장자 지원)
            image_file = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = images_dir / f"{image_name}{ext}"
                if candidate.exists():
                    image_file = candidate
                    break
            
            # 이미지가 어느 패턴 폴더에 있는지 찾기
            target_label_path = None
            
            # 모든 패턴 폴더에서 이미지 검색
            for pattern_dir in target_path.iterdir():
                if not pattern_dir.is_dir():
                    continue
                
                # 이미지 파일 찾기
                if image_file:
                    target_image = pattern_dir / image_file.name
                else:
                    # 라벨만 있고 이미지는 없는 경우
                    target_image = None
                    for ext in [".jpg", ".png", ".jpeg"]:
                        candidate = pattern_dir / f"{image_name}{ext}"
                        if candidate.exists():
                            target_image = candidate
                            break
                
                if target_image and target_image.exists():
                    # 라벨 파일 복사
                    target_label_path = pattern_dir / label_file.name
                    shutil.copy(label_file, target_label_path)
                    stats["labels_copied"] += 1
                    
                    # 이미지 복사 (옵션)
                    if copy_images and image_file and image_file.exists():
                        shutil.copy(image_file, target_image)
                        stats["images_copied"] += 1
                    
                    break
            
            if target_label_path is None:
                stats["labels_not_found"] += 1
                print(f"\n경고: 이미지를 찾을 수 없습니다: {image_name}")
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("통합 완료 통계")
    print("=" * 60)
    print(f"복사된 라벨: {stats['labels_copied']}개")
    if copy_images:
        print(f"복사된 이미지: {stats['images_copied']}개")
    print(f"찾을 수 없는 라벨: {stats['labels_not_found']}개")
    print("=" * 60)
    
    print("\n다음 단계:")
    print("1. 라벨링 진행 상황 확인:")
    print("   python utils/check_labeling_status.py")
    print("\n2. YOLO 데이터셋 생성:")
    print("   python utils/convert_labels_to_yolo.py organize \\")
    print("       --image-dir dataset/ \\")
    print("       --labels-dir dataset/ \\")
    print("       --output-dir yolo_dataset/")
    print("\n3. 학습:")
    print("   python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100")


def validate_roboflow_labels(roboflow_dir: str):
    """Roboflow 라벨 파일 형식 검증"""
    roboflow_path = Path(roboflow_dir)
    
    train_labels = roboflow_path / "train" / "labels"
    if not train_labels.exists():
        print(f"⚠️  train/labels 폴더를 찾을 수 없습니다: {train_labels}")
        return False
    
    # 샘플 라벨 파일 확인
    sample_labels = list(train_labels.glob("*.txt"))[:5]
    
    if not sample_labels:
        print("⚠️  라벨 파일을 찾을 수 없습니다.")
        return False
    
    print("라벨 파일 형식 검증 중...")
    print()
    
    all_valid = True
    for label_file in sample_labels:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        valid = True
        for i, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"❌ {label_file.name} (라인 {i}): 형식 오류 - 5개 값이 필요함")
                valid = False
                all_valid = False
                continue
            
            try:
                class_id = int(parts[0])
                center_x, center_y = float(parts[1]), float(parts[2])
                width, height = float(parts[3]), float(parts[4])
                
                # 좌표 범위 검증
                if not (0 <= center_x <= 1 and 0 <= center_y <= 1):
                    print(f"⚠️  {label_file.name} (라인 {i}): 중심 좌표가 0~1 범위를 벗어남")
                    valid = False
                if not (0 < width <= 1 and 0 < height <= 1):
                    print(f"⚠️  {label_file.name} (라인 {i}): 크기가 0~1 범위를 벗어남")
                    valid = False
                
            except ValueError:
                print(f"❌ {label_file.name} (라인 {i}): 숫자 형식 오류")
                valid = False
                all_valid = False
        
        if valid:
            print(f"✅ {label_file.name}: 형식 정상")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Roboflow 라벨 파일을 현재 프로젝트에 통합"
    )
    parser.add_argument(
        "roboflow_dir",
        type=str,
        help="Roboflow에서 다운로드한 데이터셋 디렉터리"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="dataset",
        help="대상 dataset 디렉터리 (기본: dataset)"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="이미지도 복사할지 여부 (기본: 라벨만 복사)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="라벨 파일 형식만 검증"
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        valid = validate_roboflow_labels(args.roboflow_dir)
        if valid:
            print("\n✅ 모든 라벨 파일이 올바른 형식입니다!")
        else:
            print("\n❌ 일부 라벨 파일에 문제가 있습니다.")
        return
    
    import_roboflow_labels(
        args.roboflow_dir,
        args.target_dir,
        args.copy_images
    )


if __name__ == "__main__":
    main()


