import torch
import numpy as np
from typing import Dict, List, Tuple
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from panther_metrics import PANTHERMetrics


class MetricsCalculator:
    """세그멘테이션 메트릭 계산 클래스"""
    
    def __init__(self, num_classes: int, ignore_background: bool = True):
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.eps = 1e-8
        
        # PANTHER 공식 평가기 초기화
        self.panther_evaluator = PANTHERMetrics(num_classes, ignore_background)
    
    def dice_coefficient(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Dice coefficient 계산"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice
    
    def iou_score(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """IoU (Intersection over Union) 계산"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou
    
    def sensitivity_recall(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Sensitivity (Recall) 계산"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        true_positive = (pred * target).sum()
        false_negative = ((1 - pred) * target).sum()
        
        sensitivity = (true_positive + smooth) / (true_positive + false_negative + smooth)
        return sensitivity
    
    def specificity(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Specificity 계산"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        true_negative = ((1 - pred) * (1 - target)).sum()
        false_positive = (pred * (1 - target)).sum()
        
        specificity = (true_negative + smooth) / (true_negative + false_positive + smooth)
        return specificity
    
    def precision(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Precision 계산"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        true_positive = (pred * target).sum()
        false_positive = (pred * (1 - target)).sum()
        
        precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
        return precision
    
    def f1_score(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """F1 Score 계산"""
        precision = self.precision(pred, target, smooth)
        recall = self.sensitivity_recall(pred, target, smooth)
        
        f1 = (2 * precision * recall) / (precision + recall + self.eps)
        return f1
    
    def hausdorff_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Hausdorff Distance 계산 (numpy 배열 입력)"""
        if pred.sum() == 0 and target.sum() == 0:
            return 0.0
        if pred.sum() == 0 or target.sum() == 0:
            return float('inf')
        
        # 경계 추출
        pred_points = np.column_stack(np.where(pred))
        target_points = np.column_stack(np.where(target))
        
        # Hausdorff distance 계산
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        
        return max(hd1, hd2)
    
    def surface_dice(self, pred: np.ndarray, target: np.ndarray, tolerance: float = 1.0) -> float:
        """Surface Dice 계산"""
        from skimage.segmentation import find_boundaries
        
        pred_surface = find_boundaries(pred, mode='inner')
        target_surface = find_boundaries(target, mode='inner')
        
        if pred_surface.sum() == 0 and target_surface.sum() == 0:
            return 1.0
        if pred_surface.sum() == 0 or target_surface.sum() == 0:
            return 0.0
        
        # 거리 맵 계산
        from scipy.ndimage import distance_transform_edt
        
        pred_dist = distance_transform_edt(~pred_surface)
        target_dist = distance_transform_edt(~target_surface)
        
        # 각 표면 점에서 가장 가까운 거리 계산
        pred_to_target = pred_dist[target_surface]
        target_to_pred = target_dist[pred_surface]
        
        # tolerance 내에 있는 점들의 비율
        pred_within_tolerance = (pred_to_target <= tolerance).sum()
        target_within_tolerance = (target_to_pred <= tolerance).sum()
        
        surface_dice = (pred_within_tolerance + target_within_tolerance) / (pred_surface.sum() + target_surface.sum())
        return surface_dice
    
    def calculate_basic_metrics(self, pred_logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """기본 메트릭만 빠르게 계산 (PANTHER 메트릭 제외)"""
        # Softmax 적용 후 argmax로 예측 클래스 얻기
        pred_probs = F.softmax(pred_logits, dim=1)
        pred_classes = torch.argmax(pred_probs, dim=1)
        
        metrics = {}
        
        # 각 클래스별로 기본 메트릭만 계산
        start_class = 1 if self.ignore_background else 0
        
        for class_idx in range(start_class, self.num_classes):
            # 이진 마스크 생성
            pred_binary = (pred_classes == class_idx).float()
            target_binary = (target == class_idx).float()
            
            # 기본 메트릭 계산
            dice = self.dice_coefficient(pred_binary, target_binary).item()
            iou = self.iou_score(pred_binary, target_binary).item()
            sensitivity = self.sensitivity_recall(pred_binary, target_binary).item()
            specificity_score = self.specificity(pred_binary, target_binary).item()
            precision_score = self.precision(pred_binary, target_binary).item()
            f1 = self.f1_score(pred_binary, target_binary).item()
            
            metrics[f'dice_class_{class_idx}'] = dice
            metrics[f'iou_class_{class_idx}'] = iou
            metrics[f'sensitivity_class_{class_idx}'] = sensitivity
            metrics[f'specificity_class_{class_idx}'] = specificity_score
            metrics[f'precision_class_{class_idx}'] = precision_score
            metrics[f'f1_class_{class_idx}'] = f1
        
        # 평균 메트릭 계산
        if start_class < self.num_classes:
            dice_scores = [metrics[f'dice_class_{i}'] for i in range(start_class, self.num_classes)]
            iou_scores = [metrics[f'iou_class_{i}'] for i in range(start_class, self.num_classes)]
            
            metrics['mean_dice'] = np.mean(dice_scores)
            metrics['mean_iou'] = np.mean(iou_scores)
        
        return metrics
    
    def calculate_all_metrics(self, pred_logits: torch.Tensor, target: torch.Tensor, 
                            spacing: Tuple[float, float, float] = None) -> Dict[str, float]:
        """모든 메트릭 계산 (PANTHER 공식 평가 포함)"""
        # Softmax 적용 후 argmax로 예측 클래스 얻기
        pred_probs = F.softmax(pred_logits, dim=1)
        pred_classes = torch.argmax(pred_probs, dim=1)
        
        metrics = {}
        
        # 각 클래스별로 메트릭 계산
        start_class = 1 if self.ignore_background else 0
        
        for class_idx in range(start_class, self.num_classes):
            pred_binary = (pred_classes == class_idx).float()
            target_binary = (target == class_idx).float()
            
            # 기본 메트릭들 (PyTorch 버전)
            dice = self.dice_coefficient(pred_binary, target_binary).item()
            iou = self.iou_score(pred_binary, target_binary).item()
            sensitivity = self.sensitivity_recall(pred_binary, target_binary).item()
            specificity_score = self.specificity(pred_binary, target_binary).item()
            precision_score = self.precision(pred_binary, target_binary).item()
            f1 = self.f1_score(pred_binary, target_binary).item()
            
            metrics[f'dice_class_{class_idx}'] = dice
            metrics[f'iou_class_{class_idx}'] = iou
            metrics[f'sensitivity_class_{class_idx}'] = sensitivity
            metrics[f'specificity_class_{class_idx}'] = specificity_score
            metrics[f'precision_class_{class_idx}'] = precision_score
            metrics[f'f1_class_{class_idx}'] = f1
            
            # PANTHER 공식 평가 메트릭 (첫 번째 샘플에 대해서만)
            if pred_binary.shape[0] == 1:  # 배치 크기가 1인 경우에만
                pred_np = pred_binary[0].cpu().numpy().astype(np.int32)
                target_np = target_binary[0].cpu().numpy().astype(np.int32)
                
                try:
                    # PANTHER 공식 메트릭 계산
                    panther_metrics = self.panther_evaluator.calculate_all_metrics(
                        pred_np, target_np, spacing
                    )
                    
                    # PANTHER 메트릭을 클래스별로 추가
                    for key, value in panther_metrics.items():
                        if key not in ['pred_components', 'target_components', 'true_positives', 
                                      'false_positives', 'false_negatives']:
                            metrics[f'panther_{key}_class_{class_idx}'] = value
                
                except Exception as e:
                    print(f"Warning: PANTHER metrics calculation failed: {e}")
                    # 기본값으로 대체
                    metrics[f'panther_hausdorff_distance_95_class_{class_idx}'] = 999.0
                    metrics[f'panther_average_surface_distance_class_{class_idx}'] = 999.0
                    metrics[f'panther_volume_similarity_class_{class_idx}'] = 0.0
        
        # 평균 메트릭 계산
        if self.num_classes > 2 or not self.ignore_background:
            num_valid_classes = self.num_classes - (1 if self.ignore_background else 0)
            
            for metric_name in ['dice', 'iou', 'sensitivity', 'specificity', 'precision', 'f1']:
                class_values = [metrics[f'{metric_name}_class_{i}'] 
                               for i in range(start_class, self.num_classes)]
                metrics[f'mean_{metric_name}'] = np.mean(class_values)
            
            # PANTHER 평균 메트릭도 계산
            panther_metric_names = ['panther_dice_coefficient', 'panther_iou_score', 
                                   'panther_sensitivity', 'panther_specificity', 'panther_precision',
                                   'panther_hausdorff_distance_95', 'panther_average_surface_distance',
                                   'panther_volume_similarity']
            
            for metric_name in panther_metric_names:
                class_values = []
                for i in range(start_class, self.num_classes):
                    key = f'{metric_name}_class_{i}'
                    if key in metrics and not np.isinf(metrics[key]) and not np.isnan(metrics[key]):
                        class_values.append(metrics[key])
                
                if class_values:
                    metrics[f'mean_{metric_name}'] = np.mean(class_values)
        
        return metrics


class DiceLoss(torch.nn.Module):
    """Dice Loss"""
    
    def __init__(self, num_classes: int, ignore_background: bool = True, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        
        # 라벨 값 범위 검증 및 클리핑
        target = torch.clamp(target, 0, self.num_classes - 1)
        
        # One-hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        dice_loss = 0.0
        start_class = 1 if self.ignore_background else 0
        
        for class_idx in range(start_class, self.num_classes):
            pred_class = pred[:, class_idx]
            target_class = target_one_hot[:, class_idx]
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice)
        
        num_classes = self.num_classes - (1 if self.ignore_background else 0)
        return dice_loss / num_classes


class FocalLoss(torch.nn.Module):
    """Focal Loss"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 라벨 값 범위 검증 및 클리핑
        target = torch.clamp(target, 0, pred.size(1) - 1)
        
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(torch.nn.Module):
    """Combined Loss (CE + Dice + Focal)"""
    
    def __init__(self, 
                 num_classes: int,
                 ce_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 focal_weight: float = 0.5,
                 ignore_background: bool = True):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes, ignore_background)
        self.focal_loss = FocalLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 라벨 값 범위 검증 및 클리핑
        target = torch.clamp(target, 0, pred.size(1) - 1)
        
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        total_loss = (self.ce_weight * ce + 
                     self.dice_weight * dice + 
                     self.focal_weight * focal)
        
        return total_loss


if __name__ == "__main__":
    # 메트릭 계산기 테스트
    num_classes = 2
    metrics_calc = MetricsCalculator(num_classes, ignore_background=True)
    
    # 가짜 데이터 생성
    pred_logits = torch.randn(2, num_classes, 32, 32, 32)
    target = torch.randint(0, num_classes, (2, 32, 32, 32))
    
    # 메트릭 계산
    metrics = metrics_calc.calculate_all_metrics(pred_logits[0:1], target[0:1])
    
    print("Calculated metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 손실 함수 테스트
    print("\nTesting loss functions:")
    
    dice_loss = DiceLoss(num_classes)
    focal_loss = FocalLoss()
    combined_loss = CombinedLoss(num_classes)
    
    dice_val = dice_loss(pred_logits, target)
    focal_val = focal_loss(pred_logits, target)
    combined_val = combined_loss(pred_logits, target)
    
    print(f"Dice Loss: {dice_val:.4f}")
    print(f"Focal Loss: {focal_val:.4f}")
    print(f"Combined Loss: {combined_val:.4f}")
