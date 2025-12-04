"""
패턴 분류 모델 학습 스크립트
Wafer-map 기반 패턴 분류 모델 학습
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.pattern_classifier import SimplePatternClassifier
import config


class WaferMapDataset(Dataset):
    """Wafer-map 이미지 데이터셋"""
    
    def __init__(self, data_dir: Path, transform=None):
        """
        Args:
            data_dir: 데이터셋 루트 디렉터리 (각 클래스별 하위 디렉터리 포함)
            transform: 이미지 변환 (전처리)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: i for i, cls in enumerate(config.PATTERN_CLASSES)}
        
        # 패턴명 매핑 (대소문자, 공백 처리)
        pattern_mapping = {
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
        
        # 각 클래스 디렉터리에서 이미지 수집
        for class_name in config.PATTERN_CLASSES:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        # 패턴명 매핑으로 디렉터리명 변환 처리
        for dir_path in self.data_dir.iterdir():
            if not dir_path.is_dir():
                continue
            
            dir_name = dir_path.name
            # 매핑된 패턴명으로 변환
            mapped_name = pattern_mapping.get(dir_name, None)
            if mapped_name and mapped_name in self.class_to_idx:
                class_idx = self.class_to_idx[mapped_name]
                for img_path in dir_path.glob("*.png"):
                    self.samples.append((img_path, class_idx))
                for img_path in dir_path.glob("*.jpg"):
                    self.samples.append((img_path, class_idx))
        
        print(f"데이터셋 크기: {len(self.samples)}개")
        for class_name, idx in self.class_to_idx.items():
            count = sum(1 for _, label in self.samples if label == idx)
            print(f"  {class_name}: {count}개")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 이미지 로드 (그레이스케일)
        img = Image.open(img_path).convert('L')
        
        # NumPy 배열로 변환
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # 텐서로 변환
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, H, W]
        
        # 리사이즈 (필요시)
        if img_tensor.shape[1] != config.GRID_HEIGHT or img_tensor.shape[2] != config.GRID_WIDTH:
            resize_transform = transforms.Resize((config.GRID_HEIGHT, config.GRID_WIDTH))
            img_tensor = resize_transform(img_tensor)
        
        return img_tensor, label


def train_pattern_classifier(
    data_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = None,
    val_split: float = 0.2
):
    """
    패턴 분류 모델 학습
    
    Args:
        data_dir: 데이터셋 디렉터리 경로
        epochs: 학습 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
        device: 디바이스 ("cpu", "cuda")
        val_split: 검증 데이터 비율
    """
    # 디바이스 설정
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"디바이스: {device}")
    
    # 데이터셋 로드
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"데이터셋 디렉터리를 찾을 수 없습니다: {data_dir}")
    
    full_dataset = WaferMapDataset(data_path)
    
    if len(full_dataset) == 0:
        raise ValueError("데이터셋이 비어있습니다. 각 클래스별로 이미지가 있어야 합니다.")
    
    # Train/Val 분할
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=val_split,
        shuffle=True,
        random_state=42
    )
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"학습 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    
    # 모델 초기화
    model = SimplePatternClassifier(num_classes=len(config.PATTERN_CLASSES))
    model.to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 학습 루프
    best_val_acc = 0.0
    
    print(f"\n학습 시작: {epochs} epochs")
    print("=" * 60)
    
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # 통계
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 60)
        
        # 최적 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            weights_dir = Path("weights")
            weights_dir.mkdir(exist_ok=True)
            weights_path = weights_dir / "pattern_classifier.pt"
            torch.save(model.state_dict(), weights_path)
            print(f"  ✅ 최적 모델 저장: {weights_path} (Val Acc: {val_acc:.2f}%)")
    
    print(f"\n✅ 학습 완료!")
    print(f"최적 검증 정확도: {best_val_acc:.2f}%")
    print(f"가중치 저장 위치: weights/pattern_classifier.pt")
    print(f"GUI 애플리케이션에서 자동으로 사용됩니다.")


def main():
    parser = argparse.ArgumentParser(description="패턴 분류 모델 학습")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="데이터셋 디렉터리 경로 (각 클래스별 하위 디렉터리 포함)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="학습 에포크 수 (기본: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="배치 크기 (기본: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="학습률 (기본: 0.001)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="디바이스 (기본: 자동 선택)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="검증 데이터 비율 (기본: 0.2)"
    )
    
    args = parser.parse_args()
    
    train_pattern_classifier(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        val_split=args.val_split
    )


if __name__ == "__main__":
    main()

