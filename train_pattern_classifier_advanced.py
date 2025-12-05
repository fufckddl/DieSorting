"""
고급 패턴 분류 모델 학습 스크립트
- 더 깊은 모델 아키텍처
- Transfer Learning (ResNet 기반)
- 고급 Data Augmentation (Mixup, CutMix)
- 더 정교한 학습 전략
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import RandomErasing
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter
import random

from models.pattern_classifier import SimplePatternClassifier
import config


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ImprovedPatternClassifier(nn.Module):
    """
    개선된 패턴 분류 모델
    - 더 깊은 네트워크
    - Batch Normalization
    - Dropout
    """
    def __init__(self, num_classes: int = len(config.PATTERN_CLASSES)):
        super().__init__()
        
        # 더 깊은 CNN 구조
        self.features = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # 두 번째 블록
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # 세 번째 블록
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # 네 번째 블록
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetPatternClassifier(nn.Module):
    """
    ResNet 기반 패턴 분류 모델 (Transfer Learning)
    """
    def __init__(self, num_classes: int = len(config.PATTERN_CLASSES)):
        super().__init__()
        
        # 사전학습된 ResNet18 로드
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # 첫 번째 레이어를 1채널 입력으로 수정
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
        
        # 사전학습 가중치 초기화 (1채널 변환 제외)
        self._initialize_weights()
    
    def _initialize_weights(self):
        # 첫 번째 레이어는 랜덤 초기화
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        # 나머지는 사전학습 가중치 사용 (이미 로드됨)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class WaferMapDatasetWithSplits(Dataset):
    """Wafer-map 이미지 데이터셋 (train/valid/test 분할 지원)"""
    
    def __init__(self, data_dir: Path, split: str = "train", transform=None, use_mixup: bool = False, use_cutmix: bool = False):
        """
        Args:
            data_dir: 데이터셋 루트 디렉터리
            split: "train", "valid", "test" 중 하나
            transform: 이미지 변환 (전처리)
            use_mixup: Mixup augmentation 사용 여부
            use_cutmix: CutMix augmentation 사용 여부
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_mixup = use_mixup and split == "train"
        self.use_cutmix = use_cutmix and split == "train"
        self.samples = []
        self.class_to_idx = {cls: i for i, cls in enumerate(config.PATTERN_CLASSES)}
        
        # 패턴명 매핑
        pattern_mapping = {
            "Center": "center",
            "Local": "local",
            "Edge Local": "edge-local",
            "DOUNT-TEST": "donut",
            "EDGE RING": "edge-ring",
            "SCRATCH": "scratch",
            "near full": "near-full",
            "nearfull": "near-full",
            "none": "none"
        }
        
        # 각 클래스 디렉터리에서 split별 이미지 수집
        for class_name in config.PATTERN_CLASSES:
            class_dir = self.data_dir / class_name / split
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
            mapped_name = pattern_mapping.get(dir_name, None)
            if mapped_name and mapped_name in self.class_to_idx:
                class_idx = self.class_to_idx[mapped_name]
                split_dir = dir_path / split
                if split_dir.exists():
                    for img_path in split_dir.glob("*.png"):
                        self.samples.append((img_path, class_idx))
                    for img_path in split_dir.glob("*.jpg"):
                        self.samples.append((img_path, class_idx))
                elif split == "train":
                    if dir_path.is_dir():
                        for img_path in dir_path.glob("*.png"):
                            self.samples.append((img_path, class_idx))
                        for img_path in dir_path.glob("*.jpg"):
                            self.samples.append((img_path, class_idx))
    
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
        
        # Data augmentation (train split만)
        if self.split == "train" and self.transform is not None:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, label


def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    
    # 랜덤 박스 영역 선택
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # lambda 조정
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss 계산"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_class_weights(dataset, boost_low_performance=True):
    """클래스 가중치 계산 (균등 가중치)
    
    Args:
        dataset: 학습 데이터셋
        boost_low_performance: 사용하지 않음 (호환성을 위해 유지)
    """
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    total = len(labels)
    num_classes = len(config.PATTERN_CLASSES)
    
    # 데이터셋 분포에 따른 균등 가중치 계산
    weights = torch.ones(num_classes)
    for class_idx, count in class_counts.items():
        if count > 0:
            # 역빈도 가중치: 적은 클래스에 높은 가중치
            weights[class_idx] = total / (num_classes * count)
    
    # 가중치 정규화 (모든 클래스가 균등하게 처리되도록)
    weights = weights / weights.sum() * num_classes
    
    return weights


def train_pattern_classifier_advanced(
    data_dir: str,
    model_type: str = "improved",  # "improved" or "resnet"
    epochs: int = 200,
    batch_size: int = 24,
    learning_rate: float = 0.001,
    device: str = None,
    early_stopping_patience: int = 25,
    use_focal_loss: bool = True,
    use_class_weights: bool = True,
    use_mixup: bool = True,
    use_cutmix: bool = True,
    mixup_alpha: float = 0.4,
    cutmix_alpha: float = 1.0
):
    """
    고급 패턴 분류 모델 학습
    
    Args:
        data_dir: 데이터셋 디렉터리 경로
        model_type: 모델 타입 ("improved" or "resnet")
        epochs: 학습 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
        device: 디바이스 ("cpu", "cuda")
        early_stopping_patience: Early stopping patience
        use_focal_loss: Focal Loss 사용 여부
        use_class_weights: Class Weight 사용 여부
        use_mixup: Mixup augmentation 사용 여부
        use_cutmix: CutMix augmentation 사용 여부
        mixup_alpha: Mixup alpha 파라미터
        cutmix_alpha: CutMix alpha 파라미터
    """
    # 디바이스 설정
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"디바이스: {device}")
    print("=" * 60)
    print("고급 학습 설정:")
    print(f"  모델 타입: {model_type}")
    print(f"  Focal Loss: {use_focal_loss}")
    print(f"  Class Weights: {use_class_weights}")
    print(f"  Mixup: {use_mixup}")
    print(f"  CutMix: {use_cutmix}")
    print("=" * 60)
    
    # 데이터셋 로드
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"데이터셋 디렉터리를 찾을 수 없습니다: {data_dir}")
    
    # 균형잡힌 Data augmentation (이전 성공 설정 기반)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # 15도 (적절한 회전)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 보수적 변환
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 보수적 밝기/대비 조정
    ])
    
    # Train/Valid/Test 데이터셋 로드
    train_dataset = WaferMapDatasetWithSplits(
        data_path, split="train", transform=train_transform,
        use_mixup=use_mixup, use_cutmix=use_cutmix
    )
    valid_dataset = WaferMapDatasetWithSplits(data_path, split="valid")
    test_dataset = WaferMapDatasetWithSplits(data_path, split="test")
    
    if len(train_dataset) == 0:
        raise ValueError("학습 데이터셋이 비어있습니다.")
    
    # 클래스 가중치 계산 (균등 가중치)
    class_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(train_dataset, boost_low_performance=True)
        class_weights = class_weights.to(device)
        print(f"\n클래스 가중치 (데이터셋 분포 기반 균등 가중치):")
        for i, class_name in enumerate(config.PATTERN_CLASSES):
            print(f"  {class_name:15s}: {class_weights[i]:.4f}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n학습 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(valid_dataset)}개")
    print(f"테스트 데이터: {len(test_dataset)}개")
    
    # 모델 초기화
    if model_type == "resnet":
        model = ResNetPatternClassifier(num_classes=len(config.PATTERN_CLASSES))
        print("\n모델: ResNet18 기반 (Transfer Learning)")
    else:
        model = ImprovedPatternClassifier(num_classes=len(config.PATTERN_CLASSES))
        print("\n모델: 개선된 CNN (더 깊은 네트워크)")
    
    model.to(device)
    
    # 손실 함수 설정 (균형잡힌 gamma 값)
    if use_focal_loss:
        # gamma를 3.0으로 설정 (이전 성공 설정)
        criterion = FocalLoss(alpha=class_weights, gamma=3.0)
        print(f"손실 함수: Focal Loss (gamma=3.0)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights if use_class_weights else None)
        print(f"손실 함수: CrossEntropyLoss")
    
    # 옵티마이저 (AdamW 사용 - 더 나은 정규화)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler (OneCycleLR 사용 - 더 빠른 수렴)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup (이전 성공 설정)
        anneal_strategy='cos'
    )
    
    # 학습 루프
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n학습 시작: {epochs} epochs")
    print("=" * 60)
    
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            images = images.to(device)
            labels = labels.to(device)
            
            # Mixup 또는 CutMix 적용 (균형잡힌 확률)
            # 모든 클래스에 동일한 확률 적용 (과도한 부스팅 방지)
            aug_prob = 0.5  # 50% 확률 (균형잡힌 접근)
            
            if use_mixup and random.random() < aug_prob:
                images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            elif use_cutmix and random.random() < aug_prob:
                images, y_a, y_b, lam = cutmix_data(images, labels, cutmix_alpha)
            else:
                y_a, y_b = labels, labels
                lam = 1.0
            
            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            
            # Loss 계산
            if lam != 1.0:
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 통계 (원본 라벨 기준)
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
            for images, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
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
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(valid_loader):.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        print("-" * 60)
        
        # 최적 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            weights_dir = Path("weights")
            weights_dir.mkdir(exist_ok=True)
            weights_path = weights_dir / "pattern_classifier.pt"
            torch.save(model.state_dict(), weights_path)
            print(f"  ✅ 최적 모델 저장: {weights_path} (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping: {early_stopping_patience} epochs 동안 개선 없음")
                print(f"최적 모델은 Epoch {best_epoch}에서 저장됨 (Val Acc: {best_val_acc:.2f}%)")
                break
    
    # 테스트 평가
    print("\n" + "=" * 60)
    print("테스트 데이터 평가")
    print("=" * 60)
    
    # 최적 모델 로드
    weights_path = Path("weights") / "pattern_classifier.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"최적 모델 로드: {weights_path}")
    
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    # 클래스별 정확도 계산
    num_classes = len(config.PATTERN_CLASSES)
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test 평가"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # 클래스별 통계
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    test_acc = 100 * test_correct / test_total
    print(f"\n전체 Test 정확도: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    
    print(f"\n클래스별 Test 정확도:")
    for i, class_name in enumerate(config.PATTERN_CLASSES):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"  {class_name:15s}: {class_acc:6.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  {class_name:15s}: 데이터 없음")
    
    print(f"\n✅ 학습 완료!")
    print(f"최적 검증 정확도: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"테스트 정확도: {test_acc:.2f}%")
    print(f"가중치 저장 위치: weights/pattern_classifier.pt")
    print(f"GUI 애플리케이션에서 자동으로 사용됩니다.")


def main():
    parser = argparse.ArgumentParser(description="고급 패턴 분류 모델 학습")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="데이터셋 디렉터리 경로"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="improved",
        choices=["improved", "resnet"],
        help="모델 타입 (기본: improved)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="학습 에포크 수 (기본: 200)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="배치 크기 (기본: 24)"
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
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience (기본: 25)"
    )
    parser.add_argument(
        "--no-focal-loss",
        action="store_true",
        help="Focal Loss 사용 안 함"
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Class Weights 사용 안 함"
    )
    parser.add_argument(
        "--no-mixup",
        action="store_true",
        help="Mixup augmentation 사용 안 함"
    )
    parser.add_argument(
        "--no-cutmix",
        action="store_true",
        help="CutMix augmentation 사용 안 함"
    )
    
    args = parser.parse_args()
    
    train_pattern_classifier_advanced(
        data_dir=args.data_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        early_stopping_patience=args.patience,
        use_focal_loss=not args.no_focal_loss,
        use_class_weights=not args.no_class_weights,
        use_mixup=not args.no_mixup,
        use_cutmix=not args.no_cutmix
    )


if __name__ == "__main__":
    main()

