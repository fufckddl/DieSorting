"""
라벨링 진행 상황 확인 스크립트
"""

from pathlib import Path
from collections import defaultdict

def check_labeling_status(dataset_dir: str = "dataset"):
    """라벨링 진행 상황 확인"""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"❌ 데이터셋 디렉터리를 찾을 수 없습니다: {dataset_dir}")
        return
    
    print("=" * 60)
    print("라벨링 진행 상황 확인")
    print("=" * 60)
    
    stats = defaultdict(lambda: {"images": 0, "labels": 0})
    
    # 패턴별 디렉터리 확인
    for pattern_dir in sorted(dataset_path.iterdir()):
        if not pattern_dir.is_dir():
            continue
        
        pattern_name = pattern_dir.name
        
        # 이미지 파일 개수
        images = list(pattern_dir.glob("*.jpg")) + \
                 list(pattern_dir.glob("*.png")) + \
                 list(pattern_dir.glob("*.jpeg"))
        
        # 라벨 파일 개수
        labels = list(pattern_dir.glob("*.txt"))
        
        stats[pattern_name]["images"] = len(images)
        stats[pattern_name]["labels"] = len(labels)
    
    # 통계 출력
    total_images = 0
    total_labels = 0
    
    print(f"\n{'패턴':<20} {'이미지':<10} {'라벨':<10} {'진행률':<10}")
    print("-" * 60)
    
    for pattern_name in sorted(stats.keys()):
        images = stats[pattern_name]["images"]
        labels = stats[pattern_name]["labels"]
        progress = (labels / images * 100) if images > 0 else 0
        
        total_images += images
        total_labels += labels
        
        status = "✅" if labels == images else "⚠️ "
        print(f"{status} {pattern_name:<18} {images:<10} {labels:<10} {progress:>6.1f}%")
    
    print("-" * 60)
    total_progress = (total_labels / total_images * 100) if total_images > 0 else 0
    print(f"{'전체':<20} {total_images:<10} {total_labels:<10} {total_progress:>6.1f}%")
    
    # 다음 단계 안내
    print("\n" + "=" * 60)
    if total_labels == 0:
        print("⚠️  라벨링이 시작되지 않았습니다.")
        print("\n다음 단계:")
        print("1. LabelImg 실행: labelImg")
        print("2. 각 패턴 폴더에서 라벨링 작업 진행")
        print("3. 이 스크립트를 다시 실행하여 진행 상황 확인")
    elif total_labels < total_images:
        print(f"⚠️  라벨링이 진행 중입니다. ({total_labels}/{total_images} 완료)")
        print("\n라벨링을 계속 진행하세요.")
    else:
        print("✅ 모든 이미지의 라벨링이 완료되었습니다!")
        print("\n다음 단계:")
        print("python utils/convert_labels_to_yolo.py organize \\")
        print("    --image-dir dataset/ \\")
        print("    --labels-dir dataset/ \\")
        print("    --output-dir yolo_dataset/")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    check_labeling_status(dataset_dir)


