"""
YOLO Segmentation 형식 라벨을 Detection 형식으로 변환
"""

import argparse
from pathlib import Path
from tqdm import tqdm


def segmentation_to_bbox(seg_coords):
    """
    Segmentation 좌표를 bounding box로 변환
    
    Args:
        seg_coords: 정규화된 좌표 리스트 [x1, y1, x2, y2, x3, y3, ...]
    
    Returns:
        (center_x, center_y, width, height) 정규화된 좌표
    """
    if len(seg_coords) < 4:  # 최소 2개 점 필요
        return None
    
    # x, y 좌표 분리
    x_coords = [seg_coords[i] for i in range(0, len(seg_coords), 2)]
    y_coords = [seg_coords[i+1] for i in range(0, len(seg_coords)-1, 2)]
    
    # 범위 계산
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Bounding box 계산
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # 경계 체크 (0~1 범위)
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return (center_x, center_y, width, height)


def convert_label_file(input_file, output_file):
    """
    단일 라벨 파일 변환
    
    Args:
        input_file: 입력 라벨 파일 경로
        output_file: 출력 라벨 파일 경로
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    converted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 2:
            continue
        
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        
        # 좌표 개수 확인
        if len(coords) == 4:
            # 이미 detection 형식 (center_x, center_y, width, height)
            converted_lines.append(line + '\n')
        elif len(coords) >= 4 and len(coords) % 2 == 0:
            # Segmentation 형식 (다수 좌표)
            bbox = segmentation_to_bbox(coords)
            if bbox:
                center_x, center_y, width, height = bbox
                converted_lines.append(
                    f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                )
            else:
                print(f"⚠️  변환 실패 (좌표 부족): {input_file}")
        else:
            print(f"⚠️  잘못된 형식: {input_file} (좌표 개수: {len(coords)})")
            # 원본 유지
            converted_lines.append(line + '\n')
    
    # 변환된 라벨 저장
    with open(output_file, 'w') as f:
        f.writelines(converted_lines)


def convert_dataset(input_dir, output_dir, copy_images=True):
    """
    전체 데이터셋 변환
    
    Args:
        input_dir: 입력 데이터셋 디렉터리
        output_dir: 출력 데이터셋 디렉터리
        copy_images: 이미지도 복사할지 여부
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"입력 디렉터리를 찾을 수 없습니다: {input_dir}")
    
    # 출력 디렉터리 구조 생성
    splits = ['train', 'valid', 'val', 'test']
    
    for split in splits:
        input_labels = input_path / split / 'labels'
        input_images = input_path / split / 'images'
        
        if not input_labels.exists():
            continue
        
        output_labels = output_path / split / 'labels'
        output_images = output_path / split / 'images'
        
        output_labels.mkdir(parents=True, exist_ok=True)
        if copy_images:
            output_images.mkdir(parents=True, exist_ok=True)
    
    # data.yaml 복사 및 수정
    input_yaml = input_path / 'data.yaml'
    output_yaml = output_path / 'data.yaml'
    
    if input_yaml.exists():
        import yaml
        with open(input_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # 클래스 이름을 'defect'로 변경 (선택적)
        if 'names' in yaml_data and isinstance(yaml_data['names'], list):
            # 클래스 ID 0을 'defect'로 변경
            yaml_data['names'] = ['defect']
        elif 'names' in yaml_data and isinstance(yaml_data['names'], dict):
            yaml_data['names'] = {0: 'defect'}
        
        # 경로 수정
        yaml_data['train'] = '../train/images'
        yaml_data['val'] = '../valid/images' if (output_path / 'valid').exists() else '../val/images'
        if 'test' in yaml_data:
            yaml_data['test'] = '../test/images'
        
        with open(output_yaml, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    # 각 split별로 변환
    stats = {
        'total_labels': 0,
        'converted_labels': 0,
        'detection_format': 0,
        'segmentation_format': 0,
        'errors': 0
    }
    
    for split in splits:
        input_labels = input_path / split / 'labels'
        input_images = input_path / split / 'images'
        output_labels = output_path / split / 'labels'
        output_images = output_path / split / 'images'
        
        if not input_labels.exists():
            continue
        
        print(f"\n[{split}] 처리 중...")
        
        label_files = list(input_labels.glob('*.txt'))
        stats['total_labels'] += len(label_files)
        
        for label_file in tqdm(label_files, desc=f"{split} 변환"):
            try:
                output_label = output_labels / label_file.name
                
                # 라벨 파일 읽기 및 형식 확인
                with open(label_file, 'r') as f:
                    first_line = f.readline().strip()
                    parts = first_line.split()
                    coords = [float(x) for x in parts[1:]] if len(parts) > 1 else []
                
                if len(coords) == 4:
                    # 이미 detection 형식
                    stats['detection_format'] += 1
                    import shutil
                    shutil.copy(label_file, output_label)
                elif len(coords) >= 4 and len(coords) % 2 == 0:
                    # Segmentation 형식
                    stats['segmentation_format'] += 1
                    convert_label_file(label_file, output_label)
                    stats['converted_labels'] += 1
                else:
                    # 잘못된 형식
                    stats['errors'] += 1
                    print(f"⚠️  잘못된 형식: {label_file}")
                    import shutil
                    shutil.copy(label_file, output_label)  # 원본 유지
                
                # 이미지 복사
                if copy_images and input_images.exists():
                    image_name = label_file.stem
                    for ext in ['.jpg', '.png', '.jpeg']:
                        input_image = input_images / f"{image_name}{ext}"
                        if input_image.exists():
                            import shutil
                            shutil.copy(input_image, output_images / input_image.name)
                            break
                
            except Exception as e:
                stats['errors'] += 1
                print(f"❌ 오류 발생 ({label_file}): {e}")
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("변환 완료 통계")
    print("=" * 60)
    print(f"총 라벨 파일: {stats['total_labels']}개")
    print(f"Detection 형식 (그대로 유지): {stats['detection_format']}개")
    print(f"Segmentation 형식 (변환됨): {stats['segmentation_format']}개")
    print(f"변환된 라벨: {stats['converted_labels']}개")
    print(f"오류: {stats['errors']}개")
    print("=" * 60)
    
    print(f"\n✅ 변환 완료!")
    print(f"출력 위치: {output_path}")
    print(f"\n다음 단계:")
    print(f"1. 변환된 데이터셋 검증:")
    print(f"   python utils/import_roboflow_labels.py {output_path} --validate-only")
    print(f"\n2. 학습 실행:")
    print(f"   python train_yolo.py --data {output_path}/data.yaml --epochs 100")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Segmentation 형식을 Detection 형식으로 변환"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='입력 데이터셋 디렉터리 (Roboflow 형식)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='출력 데이터셋 디렉터리'
    )
    parser.add_argument(
        '--no-copy-images',
        action='store_true',
        help='이미지 복사하지 않기'
    )
    
    args = parser.parse_args()
    
    convert_dataset(
        args.input_dir,
        args.output_dir,
        copy_images=not args.no_copy_images
    )


if __name__ == '__main__':
    main()


