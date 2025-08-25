import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from typing import Tuple, List, Optional, Dict
import random
from scipy import ndimage
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PANTHERDataset(Dataset):
    """PANTHER Task1 3D Medical Image Segmentation Dataset"""
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        image_ids: List[str] = None,
        crop_size: Tuple[int, int, int] = (72, 220, 220),
        num_classes: int = 2,
        augment: bool = True,
        normalize: bool = True,
        spacing: Tuple[float, float, float] = None,
        cache_data: bool = False
    ):
        """
        Args:
            images_dir: 이미지 파일들이 있는 디렉토리 경로
            labels_dir: 라벨 파일들이 있는 디렉토리 경로
            image_ids: 사용할 이미지 ID 리스트 (None이면 모든 파일 사용)
            crop_size: 크롭할 이미지 크기 (D, H, W)
            num_classes: 클래스 개수
            augment: 데이터 증강 여부
            normalize: 정규화 여부
            spacing: 리샘플링할 spacing (None이면 원본 spacing 사용)
            cache_data: 메모리에 데이터 캐시 여부
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.augment = augment
        self.normalize = normalize
        self.spacing = spacing
        self.cache_data = cache_data
        
        # 이미지 파일 리스트 생성
        if image_ids is None:
            self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.mha')]
            self.image_ids = [f.split('_0000.mha')[0] for f in self.image_files]
        else:
            self.image_ids = image_ids
            self.image_files = [f"{id_}_0001_0000.mha" for id_ in image_ids]
        
        # 라벨 파일 리스트 생성
        self.label_files = [f"{id_}_0001.mha" for id_ in self.image_ids]
        
        # 캐시 딕셔너리
        self.cache = {} if cache_data else None
        
        print(f"Dataset initialized with {len(self.image_ids)} samples")
        
    def __len__(self):
        return len(self.image_ids)
    
    def _load_image_and_label(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """이미지와 라벨을 로드하고 전처리"""
        image_id = self.image_ids[idx]
        
        # 캐시에서 확인
        if self.cache is not None and image_id in self.cache:
            return self.cache[image_id]
        
        # 파일 경로
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        
        # 이미지와 라벨 로드
        image_sitk = sitk.ReadImage(image_path)
        label_sitk = sitk.ReadImage(label_path)
        
        # 배열로 변환 (Z, Y, X)
        image = sitk.GetArrayFromImage(image_sitk).astype(np.float16)
        label = sitk.GetArrayFromImage(label_sitk).astype(np.int64)
        
        # 라벨 값 검증 및 클리핑
        max_class = 2  # 0, 1, 2 클래스
        if label.max() > max_class:
            print(f"Warning: Label {label_path} has values > {max_class}, clipping...")
            label = np.clip(label, 0, max_class)
        
        if label.min() < 0:
            print(f"Warning: Label {label_path} has negative values, clipping...")
            label = np.clip(label, 0, max_class)
        
        # 마스크 값 재매핑: 0,1 -> 0, 2 -> 1
        label_remapped = np.zeros_like(label)
        label_remapped[label == 2] = 1  # 2를 1로
        label_remapped[(label == 0) | (label == 1)] = 0  # 0과 1을 0으로
        label = label_remapped
        
        # 리샘플링 (옵션)
        if self.spacing is not None:
            image, label = self._resample_image_label(
                image_sitk, label_sitk, self.spacing
            )
        
        # 정규화
        if self.normalize:
            image = self._normalize_image(image)
        
        # 캐시에 저장
        if self.cache is not None:
            self.cache[image_id] = (image, label)
        
        return image, label
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 정규화 (0-1 범위로)"""
        # 클리핑 (이상값 제거)
        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip(image, p1, p99)
        
        # Min-Max 정규화
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image
    
    def _resample_image_label(
        self, 
        image_sitk: sitk.Image, 
        label_sitk: sitk.Image, 
        new_spacing: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """이미지와 라벨을 새로운 spacing으로 리샘플링"""
        # 이미지 리샘플링
        original_spacing = image_sitk.GetSpacing()
        original_size = image_sitk.GetSize()
        
        new_size = [
            int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
            for i in range(3)
        ]
        
        # 이미지 리샘플링
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkLinear)
        image_resampled = resampler.Execute(image_sitk)
        
        # 라벨 리샘플링 (Nearest neighbor)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        label_resampled = resampler.Execute(label_sitk)
        
        image = sitk.GetArrayFromImage(image_resampled).astype(np.float32)
        label = sitk.GetArrayFromImage(label_resampled).astype(np.int64)
        
        return image, label
    
    def _random_crop_3d(
        self, 
        image: np.ndarray, 
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """3D 랜덤 크롭"""
        d, h, w = image.shape
        crop_d, crop_h, crop_w = self.crop_size
        
        # 크롭 크기가 원본보다 큰 경우 패딩
        if d < crop_d or h < crop_h or w < crop_w:
            pad_d = max(0, crop_d - d)
            pad_h = max(0, crop_h - h)
            pad_w = max(0, crop_w - w)
            
            image = np.pad(image, (
                (pad_d//2, pad_d - pad_d//2),
                (pad_h//2, pad_h - pad_h//2),
                (pad_w//2, pad_w - pad_w//2)
            ), mode='constant', constant_values=0)
            
            label = np.pad(label, (
                (pad_d//2, pad_d - pad_d//2),
                (pad_h//2, pad_h - pad_h//2),
                (pad_w//2, pad_w - pad_w//2)
            ), mode='constant', constant_values=0)
            
            d, h, w = image.shape
        
        # 랜덤 시작점 선택
        start_d = random.randint(0, d - crop_d)
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        
        # 크롭
        image_crop = image[start_d:start_d+crop_d, 
                          start_h:start_h+crop_h, 
                          start_w:start_w+crop_w]
        label_crop = label[start_d:start_d+crop_d, 
                          start_h:start_h+crop_h, 
                          start_w:start_w+crop_w]
        
        return image_crop, label_crop
    
    def _augment_3d(
        self, 
        image: np.ndarray, 
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """3D 데이터 증강"""
        if not self.augment:
            return image, label
        
        # 랜덤 플립
        if random.random() > 0.5:
            axis = random.choice([0, 1, 2])
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
        
        # 랜덤 회전
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            axes = random.choice([(0, 1), (0, 2), (1, 2)])
            image = ndimage.rotate(image, angle, axes=axes, reshape=False, order=1)
            label = ndimage.rotate(label, angle, axes=axes, reshape=False, order=0)
        
        # 가우시안 노이즈
        if random.random() > 0.8:
            noise = np.random.normal(0, 0.1, image.shape)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        # 밝기 조정
        if random.random() > 0.8:
            factor = random.uniform(0.8, 1.2)
            image = image * factor
            image = np.clip(image, 0, 1)
        
        return image, label
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """데이터셋 아이템 반환"""
        image, label = self._load_image_and_label(idx)
        
        # 랜덤 크롭
        image, label = self._random_crop_3d(image, label)
        
        # 데이터 증강
        image, label = self._augment_3d(image, label)
        
        # 채널 차원 추가 (1, D, H, W)
        image = np.expand_dims(image, axis=0)
        
        # 텐서로 변환
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        return {
            'image': image,
            'label': label,
            'image_id': self.image_ids[idx]
        }


def create_data_loaders(
    images_dir: str,
    labels_dir: str,
    train_ratio: float = 0.8,
    batch_size: int = 2,
    num_workers: int = 4,
    crop_size: Tuple[int, int, int] = (128, 128, 128),
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """훈련 및 검증 데이터로더 생성"""
    
    # 모든 이미지 ID 가져오기
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.mha')]
    image_ids = [f.split('_0001_0000.mha')[0] for f in image_files]
    
    # 훈련/검증 분할
    random.seed(random_seed)
    random.shuffle(image_ids)
    
    train_size = int(len(image_ids) * train_ratio)
    train_ids = image_ids[:train_size]
    val_ids = image_ids[train_size:]
    
    print(f"Total samples: {len(image_ids)}")
    print(f"Train samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")
    
    # 데이터셋 생성
    train_dataset = PANTHERDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_ids=train_ids,
        crop_size=crop_size,
        augment=True,
        normalize=True
    )
    
    val_dataset = PANTHERDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_ids=val_ids,
        crop_size=crop_size,
        augment=False,
        normalize=True
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 테스트
    images_dir = "/workspace/MICCAI_challenge/PANTHER/PANTHER_Task1/ImagesTr"
    labels_dir = "/workspace/MICCAI_challenge/PANTHER/PANTHER_Task1/LabelsTr"
    
    train_loader, val_loader = create_data_loaders(
        images_dir=images_dir,
        labels_dir=labels_dir,
        batch_size=1,
        crop_size=(96, 96, 96)
    )
    
    # 첫 번째 배치 확인
    for batch in train_loader:
        print(f"Image shape: {batch['image'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        print(f"Image ID: {batch['image_id']}")
        print(f"Label unique values: {torch.unique(batch['label'])}")
        break