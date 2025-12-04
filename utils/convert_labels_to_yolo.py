"""
라벨링 도구 결과를 YOLO 형식으로 변환하는 유틸리티
labelImg, CVAT, Roboflow 등 다양한 형식에서 YOLO 형식으로 변환
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
from tqdm import tqdm


def convert_labelimg_to_yolo(
    image_dir: str,
    xml_dir: str = None,
    output_dir: str = None,
    class_mapping: dict = None
):
    """
    LabelImg XML 형식을 YOLO 형식으로 변환
    
    Args:
        image_dir: 이미지 디렉터리
        xml_dir: XML 라벨 디렉터리 (None이면 image_dir과 동일)
        output_dir: YOLO 라벨 출력 디렉터리
        class_mapping: 클래스명 매핑 (예: {"defect": 0})
    """
    image_path = Path(image_dir)
    
    if xml_dir is None:
        xml_path = image_path
    else:
        xml_path = Path(xml_dir)
    
    if output_dir is None:
        output_path = image_path / "labels"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    if class_mapping is None:
        class_mapping = {"defect": 0}
    
    # XML 파일 찾기
    xml_files = list(xml_path.glob("*.xml"))
    
    print(f"총 {len(xml_files)}개 XML 파일 처리 중...")
    
    for xml_file in tqdm(xml_files):
        try:
            # XML 파싱
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 이미지 크기 가져오기
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            # YOLO 라벨 생성
            yolo_labels = []
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                
                # 클래스 ID 찾기
                if class_name not in class_mapping:
                    print(f"경고: 알 수 없는 클래스 '{class_name}' (무시됨)")
                    continue
                
                class_id = class_mapping[class_name]
                
                # bbox 좌표 가져오기
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # YOLO 형식으로 변환 (정규화된 좌표)
                center_x = ((xmin + xmax) / 2.0) / img_width
                center_y = ((ymin + ymax) / 2.0) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            # YOLO 라벨 파일 저장
            if yolo_labels:
                output_file = output_path / f"{xml_file.stem}.txt"
                with open(output_file, 'w') as f:
                    f.writelines(yolo_labels)
        
        except Exception as e:
            print(f"오류 발생 ({xml_file}): {e}")
            continue
    
    print(f"\n✅ 변환 완료!")
    print(f"YOLO 라벨 저장 위치: {output_path}")


def create_yolo_dataset_structure(
    source_dir: str,
    labels_dir: str,
    output_dir: str,
    train_split: float = 0.8
):
    """
    라벨링된 이미지와 라벨을 YOLO 데이터셋 구조로 정리
    
    Args:
        source_dir: 이미지 소스 디렉터리 (패턴별로 분류되어 있을 수 있음)
        labels_dir: YOLO 형식 라벨 디렉터리
        output_dir: 출력 데이터셋 디렉터리
        train_split: 학습 데이터 비율
    """
    source_path = Path(source_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉터리 구조 생성
    train_images = output_path / "train" / "images"
    train_labels = output_path / "train" / "labels"
    val_images = output_path / "val" / "images"
    val_labels = output_path / "val" / "labels"
    
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 이미지와 라벨 파일 매칭
    import random
    import shutil
    
    image_files = []
    
    # 패턴별 디렉터리에서 이미지 수집
    for pattern_dir in source_path.iterdir():
        if not pattern_dir.is_dir():
            continue
        
        # 이미지와 같은 디렉터리에 라벨이 있는 경우
        for img_file in pattern_dir.glob("*.jpg"):
            # 먼저 같은 디렉터리에서 찾기
            label_file = pattern_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                # 지정된 라벨 디렉터리에서 찾기
                label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                image_files.append((img_file, label_file))
        
        for img_file in pattern_dir.glob("*.png"):
            label_file = pattern_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                image_files.append((img_file, label_file))
        
        for img_file in pattern_dir.glob("*.jpeg"):
            label_file = pattern_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                image_files.append((img_file, label_file))
    
    print(f"총 {len(image_files)}개 이미지-라벨 쌍 발견")
    
    # Train/Val 분할
    random.seed(42)
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"학습 데이터: {len(train_files)}개")
    print(f"검증 데이터: {len(val_files)}개\n")
    
    # 파일 복사
    def copy_files(files, img_dir, label_dir, split_name):
        for img_file, label_file in tqdm(files, desc=f"{split_name} 복사 중"):
            shutil.copy(img_file, img_dir / img_file.name)
            shutil.copy(label_file, label_dir / label_file.name)
    
    copy_files(train_files, train_images, train_labels, "학습")
    copy_files(val_files, val_images, val_labels, "검증")
    
    # dataset.yaml 생성
    import yaml
    dataset_yaml = {
        'path': str(output_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'defect'},
        'nc': 1
    }
    
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ 데이터셋 준비 완료!")
    print(f"출력 위치: {output_path}")
    print(f"YAML 파일: {yaml_path}")
    print(f"\n다음 단계:")
    print(f"python train_yolo.py --data {yaml_path} --epochs 100")


def main():
    parser = argparse.ArgumentParser(
        description="라벨링 결과를 YOLO 형식으로 변환"
    )
    parser.add_argument(
        "mode",
        choices=["convert", "organize"],
        help="모드: convert (XML→YOLO 변환) 또는 organize (데이터셋 구조 정리)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="이미지 디렉터리"
    )
    parser.add_argument(
        "--xml-dir",
        type=str,
        default=None,
        help="XML 라벨 디렉터리 (convert 모드)"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="YOLO 라벨 디렉터리 (organize 모드)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="출력 디렉터리"
    )
    
    args = parser.parse_args()
    
    if args.mode == "convert":
        convert_labelimg_to_yolo(
            args.image_dir,
            args.xml_dir,
            args.output_dir
        )
    elif args.mode == "organize":
        if args.labels_dir is None:
            args.labels_dir = Path(args.image_dir) / "labels"
        create_yolo_dataset_structure(
            args.image_dir,
            args.labels_dir,
            args.output_dir or "yolo_dataset"
        )


if __name__ == "__main__":
    main()

