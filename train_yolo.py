"""
YOLOv8 검출 모델 학습 스크립트
반도체 불량 검출에 특화된 YOLO 모델 학습
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import config


def train_yolo_detector(
    data_yaml: str,
    model_size: str = "n",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = None,
    project: str = "runs/detect",
    name: str = "yolo_detector"
):
    """
    YOLO 검출 모델 학습
    
    Args:
        data_yaml: 데이터셋 YAML 파일 경로
        model_size: 모델 크기 ("n", "s", "m", "l", "x")
        epochs: 학습 에포크 수
        imgsz: 이미지 크기
        batch: 배치 크기
        device: 디바이스 ("cpu", "cuda", "0", "1", ...)
        project: 프로젝트 디렉터리
        name: 실행 이름
    """
    # 모델 초기화
    model_name = f"yolov8{model_size}.pt"
    print(f"모델 로드: {model_name}")
    model = YOLO(model_name)
    
    # 데이터셋 파일 확인
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {data_yaml}")
    
    print(f"데이터셋: {data_yaml}")
    print(f"학습 시작: {epochs} epochs, batch={batch}, imgsz={imgsz}")
    
    # 학습 실행
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        save=True,
        plots=True
    )
    
    # 학습 완료 후 best.pt를 weights 디렉터리로 복사
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    best_weights = Path(project) / name / "weights" / "best.pt"
    target_weights = weights_dir / "yolo_detector.pt"
    
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, target_weights)
        print(f"\n✅ 학습 완료!")
        print(f"최적 가중치가 저장되었습니다: {target_weights}")
        print(f"GUI 애플리케이션에서 자동으로 사용됩니다.")
    else:
        print(f"\n⚠️  최적 가중치 파일을 찾을 수 없습니다: {best_weights}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 검출 모델 학습")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="데이터셋 YAML 파일 경로 (예: dataset.yaml)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=config.YOLO_MODEL_SIZE,
        choices=["n", "s", "m", "l", "x"],
        help="모델 크기 (기본: n)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="학습 에포크 수 (기본: 100)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="이미지 크기 (기본: 640)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="배치 크기 (기본: 16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="디바이스 (기본: 자동 선택)"
    )
    
    args = parser.parse_args()
    
    train_yolo_detector(
        data_yaml=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )


if __name__ == "__main__":
    main()


