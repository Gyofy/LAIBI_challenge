"""
PANTHER Task1 공식 평가 지표 구현
DIAGNijmegen/PANTHER_baseline 기반
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
import SimpleITK as sitk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd


class PANTHERMetrics:
    """PANTHER Task1 공식 평가 메트릭 클래스"""
    
    def __init__(self, num_classes: int = 2, ignore_background: bool = True):
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.eps = 1e-8
        
        # PANTHER Task1에서 중요한 메트릭들
        self.primary_metrics = [
            'dice_coefficient',
            'iou_score', 
            'sensitivity',
            'specificity',
            'precision',
            'hausdorff_distance_95',
            'average_surface_distance',
            'volume_similarity'
        ]
    
    def dice_coefficient(self, pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
        """Dice Coefficient (DSC) 계산"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        intersection = np.sum(pred_flat * target_flat)
        union = np.sum(pred_flat) + np.sum(target_flat)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return float(dice)
    
    def iou_score(self, pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
        """Intersection over Union (IoU) 계산"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        intersection = np.sum(pred_flat * target_flat)
        union = np.sum(pred_flat) + np.sum(target_flat) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return float(iou)
    
    def sensitivity_recall(self, pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
        """Sensitivity (Recall) 계산"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        true_positive = np.sum(pred_flat * target_flat)
        false_negative = np.sum((1 - pred_flat) * target_flat)
        
        sensitivity = (true_positive + smooth) / (true_positive + false_negative + smooth)
        return float(sensitivity)
    
    def specificity(self, pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
        """Specificity 계산"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        true_negative = np.sum((1 - pred_flat) * (1 - target_flat))
        false_positive = np.sum(pred_flat * (1 - target_flat))
        
        specificity = (true_negative + smooth) / (true_negative + false_positive + smooth)
        return float(specificity)
    
    def precision(self, pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
        """Precision (PPV) 계산"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        true_positive = np.sum(pred_flat * target_flat)
        false_positive = np.sum(pred_flat * (1 - target_flat))
        
        precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
        return float(precision)
    
    def f1_score(self, pred: np.ndarray, target: np.ndarray) -> float:
        """F1 Score 계산"""
        precision = self.precision(pred, target)
        recall = self.sensitivity_recall(pred, target)
        
        f1 = (2 * precision * recall) / (precision + recall + self.eps)
        return float(f1)
    
    def hausdorff_distance_95(self, pred: np.ndarray, target: np.ndarray) -> float:
        """95% Hausdorff Distance 계산"""
        if pred.sum() == 0 and target.sum() == 0:
            return 0.0
        if pred.sum() == 0 or target.sum() == 0:
            return float('inf')
        
        # 경계 추출
        pred_surface = self._get_surface_points(pred)
        target_surface = self._get_surface_points(target)
        
        if len(pred_surface) == 0 or len(target_surface) == 0:
            return float('inf')
        
        # 모든 거리 계산
        distances_pred_to_target = []
        distances_target_to_pred = []
        
        for p_point in pred_surface:
            min_dist = np.min([np.linalg.norm(p_point - t_point) for t_point in target_surface])
            distances_pred_to_target.append(min_dist)
        
        for t_point in target_surface:
            min_dist = np.min([np.linalg.norm(t_point - p_point) for p_point in pred_surface])
            distances_target_to_pred.append(min_dist)
        
        # 95% percentile 계산
        hd95_1 = np.percentile(distances_pred_to_target, 95)
        hd95_2 = np.percentile(distances_target_to_pred, 95)
        
        return float(max(hd95_1, hd95_2))
    
    def average_surface_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Average Surface Distance (ASD) 계산"""
        if pred.sum() == 0 and target.sum() == 0:
            return 0.0
        if pred.sum() == 0 or target.sum() == 0:
            return float('inf')
        
        # 경계 추출
        pred_surface = self._get_surface_points(pred)
        target_surface = self._get_surface_points(target)
        
        if len(pred_surface) == 0 or len(target_surface) == 0:
            return float('inf')
        
        # 양방향 평균 거리 계산
        distances_pred_to_target = []
        distances_target_to_pred = []
        
        for p_point in pred_surface:
            min_dist = np.min([np.linalg.norm(p_point - t_point) for t_point in target_surface])
            distances_pred_to_target.append(min_dist)
        
        for t_point in target_surface:
            min_dist = np.min([np.linalg.norm(t_point - p_point) for p_point in pred_surface])
            distances_target_to_pred.append(min_dist)
        
        all_distances = distances_pred_to_target + distances_target_to_pred
        return float(np.mean(all_distances))
    
    def volume_similarity(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Volume Similarity 계산"""
        vol_pred = np.sum(pred)
        vol_target = np.sum(target)
        
        if vol_pred == 0 and vol_target == 0:
            return 1.0
        
        vol_sim = 1.0 - abs(vol_pred - vol_target) / (vol_pred + vol_target + self.eps)
        return float(vol_sim)
    
    def relative_volume_difference(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Relative Volume Difference 계산"""
        vol_pred = np.sum(pred)
        vol_target = np.sum(target)
        
        if vol_target == 0:
            return float('inf') if vol_pred > 0 else 0.0
        
        rvd = (vol_pred - vol_target) / vol_target
        return float(rvd)
    
    def _get_surface_points(self, mask: np.ndarray) -> np.ndarray:
        """마스크에서 표면 점들 추출"""
        # 경계 추출
        from skimage.segmentation import find_boundaries
        boundaries = find_boundaries(mask, mode='inner', background=0)
        
        # 표면 점 좌표 추출
        surface_points = np.column_stack(np.where(boundaries))
        return surface_points
    
    def connected_components_analysis(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """연결 성분 분석"""
        # 예측에서 연결 성분 찾기
        pred_labeled, pred_num_components = ndimage.label(pred)
        target_labeled, target_num_components = ndimage.label(target)
        
        # 각 예측 성분에 대해 타겟과의 겹침 확인
        true_positives = 0
        false_positives = 0
        
        for i in range(1, pred_num_components + 1):
            pred_component = (pred_labeled == i)
            overlap_with_target = np.any(pred_component & target)
            
            if overlap_with_target:
                true_positives += 1
            else:
                false_positives += 1
        
        # 놓친 타겟 성분들
        false_negatives = 0
        for i in range(1, target_num_components + 1):
            target_component = (target_labeled == i)
            overlap_with_pred = np.any(target_component & pred)
            
            if not overlap_with_pred:
                false_negatives += 1
        
        return {
            'pred_components': pred_num_components,
            'target_components': target_num_components,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'detection_sensitivity': true_positives / max(target_num_components, 1),
            'detection_precision': true_positives / max(pred_num_components, 1) if pred_num_components > 0 else 0.0
        }
    
    def calculate_all_metrics(self, pred: np.ndarray, target: np.ndarray, 
                             spacing: Optional[Tuple[float, float, float]] = None) -> Dict[str, float]:
        """모든 메트릭 계산"""
        
        # 입력 검증
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        
        # 이진 마스크로 변환
        pred_binary = (pred > 0.5).astype(np.int32) if pred.dtype == np.float32 else pred.astype(np.int32)
        target_binary = target.astype(np.int32)
        
        metrics = {}
        
        # 기본 메트릭들
        metrics['dice_coefficient'] = self.dice_coefficient(pred_binary, target_binary)
        metrics['iou_score'] = self.iou_score(pred_binary, target_binary)
        metrics['sensitivity'] = self.sensitivity_recall(pred_binary, target_binary)
        metrics['specificity'] = self.specificity(pred_binary, target_binary)
        metrics['precision'] = self.precision(pred_binary, target_binary)
        metrics['f1_score'] = self.f1_score(pred_binary, target_binary)
        
        # 볼륨 관련 메트릭
        metrics['volume_similarity'] = self.volume_similarity(pred_binary, target_binary)
        metrics['relative_volume_difference'] = self.relative_volume_difference(pred_binary, target_binary)
        
        # 거리 기반 메트릭 (계산이 오래 걸릴 수 있음)
        try:
            if spacing is not None:
                # 실제 물리적 간격을 고려한 거리 계산
                pred_scaled = self._apply_spacing(pred_binary, spacing)
                target_scaled = self._apply_spacing(target_binary, spacing)
                metrics['hausdorff_distance_95'] = self.hausdorff_distance_95(pred_scaled, target_scaled)
                metrics['average_surface_distance'] = self.average_surface_distance(pred_scaled, target_scaled)
            else:
                metrics['hausdorff_distance_95'] = self.hausdorff_distance_95(pred_binary, target_binary)
                metrics['average_surface_distance'] = self.average_surface_distance(pred_binary, target_binary)
        except Exception as e:
            print(f"Warning: Distance metric calculation failed: {e}")
            metrics['hausdorff_distance_95'] = float('inf')
            metrics['average_surface_distance'] = float('inf')
        
        # 연결 성분 분석
        try:
            cc_metrics = self.connected_components_analysis(pred_binary, target_binary)
            metrics.update(cc_metrics)
        except Exception as e:
            print(f"Warning: Connected components analysis failed: {e}")
        
        return metrics
    
    def _apply_spacing(self, mask: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
        """물리적 간격을 고려한 마스크 변환"""
        # 실제 구현에서는 spacing을 고려한 좌표 변환이 필요
        # 여기서는 단순화된 버전
        return mask
    
    def evaluate_case(self, pred_path: str, target_path: str) -> Dict[str, float]:
        """단일 케이스에 대한 평가"""
        # 파일 로드
        if pred_path.endswith('.nii.gz') or pred_path.endswith('.mha'):
            pred_sitk = sitk.ReadImage(pred_path)
            pred_array = sitk.GetArrayFromImage(pred_sitk)
            spacing = pred_sitk.GetSpacing()
        else:
            pred_array = np.load(pred_path)
            spacing = None
        
        if target_path.endswith('.nii.gz') or target_path.endswith('.mha'):
            target_sitk = sitk.ReadImage(target_path)
            target_array = sitk.GetArrayFromImage(target_sitk)
        else:
            target_array = np.load(target_path)
        
        return self.calculate_all_metrics(pred_array, target_array, spacing)
    
    def evaluate_dataset(self, pred_paths: List[str], target_paths: List[str], 
                        case_ids: Optional[List[str]] = None) -> Dict[str, Union[float, List[float]]]:
        """전체 데이터셋에 대한 평가"""
        assert len(pred_paths) == len(target_paths), "Number of predictions and targets must match"
        
        if case_ids is None:
            case_ids = [f"case_{i:04d}" for i in range(len(pred_paths))]
        
        all_metrics = []
        failed_cases = []
        
        for i, (pred_path, target_path, case_id) in enumerate(zip(pred_paths, target_paths, case_ids)):
            try:
                case_metrics = self.evaluate_case(pred_path, target_path)
                case_metrics['case_id'] = case_id
                all_metrics.append(case_metrics)
                print(f"Evaluated {case_id}: Dice={case_metrics['dice_coefficient']:.4f}")
            except Exception as e:
                print(f"Failed to evaluate {case_id}: {e}")
                failed_cases.append(case_id)
        
        if not all_metrics:
            raise ValueError("No cases were successfully evaluated")
        
        # 평균 메트릭 계산
        aggregated_metrics = {}
        metric_names = [key for key in all_metrics[0].keys() if key != 'case_id' and isinstance(all_metrics[0][key], (int, float))]
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics if not np.isinf(m[metric_name]) and not np.isnan(m[metric_name])]
            if values:
                aggregated_metrics[f'mean_{metric_name}'] = np.mean(values)
                aggregated_metrics[f'std_{metric_name}'] = np.std(values)
                aggregated_metrics[f'median_{metric_name}'] = np.median(values)
                aggregated_metrics[f'min_{metric_name}'] = np.min(values)
                aggregated_metrics[f'max_{metric_name}'] = np.max(values)
        
        # 개별 케이스 결과도 포함
        aggregated_metrics['individual_results'] = all_metrics
        aggregated_metrics['failed_cases'] = failed_cases
        aggregated_metrics['num_successful_cases'] = len(all_metrics)
        aggregated_metrics['num_failed_cases'] = len(failed_cases)
        
        return aggregated_metrics
    
    def save_results(self, results: Dict, output_path: str):
        """결과를 CSV 파일로 저장"""
        if 'individual_results' in results:
            # 개별 결과를 DataFrame으로 변환
            df = pd.DataFrame(results['individual_results'])
            df.to_csv(output_path.replace('.json', '_individual.csv'), index=False)
            
            # 요약 통계를 별도 파일로 저장
            summary_results = {k: v for k, v in results.items() if k != 'individual_results'}
            summary_df = pd.DataFrame([summary_results])
            summary_df.to_csv(output_path.replace('.json', '_summary.csv'), index=False)
        
        # JSON 형태로도 저장
        import json
        with open(output_path, 'w') as f:
            # JSON serializable하게 변환
            json_results = {}
            for k, v in results.items():
                if isinstance(v, (list, dict)):
                    json_results[k] = v
                elif isinstance(v, np.ndarray):
                    json_results[k] = v.tolist()
                elif np.isnan(v) or np.isinf(v):
                    json_results[k] = None
                else:
                    json_results[k] = float(v) if isinstance(v, (np.integer, np.floating)) else v
            
            json.dump(json_results, f, indent=2)


def evaluate_panther_predictions(pred_dir: str, target_dir: str, output_path: str):
    """PANTHER Task1 예측 결과 평가 메인 함수"""
    import os
    from glob import glob
    
    # 평가기 초기화
    evaluator = PANTHERMetrics(num_classes=2, ignore_background=True)
    
    # 예측 파일들 찾기
    pred_files = sorted(glob(os.path.join(pred_dir, "*.mha")))
    
    # 대응하는 타겟 파일들 찾기
    target_files = []
    case_ids = []
    
    for pred_file in pred_files:
        # 파일명에서 케이스 ID 추출
        case_id = os.path.basename(pred_file).replace('.mha', '').replace('_prediction', '')
        case_ids.append(case_id)
        
        # 대응하는 타겟 파일 찾기
        possible_target_names = [
            f"{case_id}.mha",
            f"{case_id}_label.mha", 
            f"{case_id}_0001.mha",
            f"{case_id.replace('_0001_0000', '_0001')}.mha"
        ]
        
        target_file = None
        for target_name in possible_target_names:
            target_path = os.path.join(target_dir, target_name)
            if os.path.exists(target_path):
                target_file = target_path
                break
        
        if target_file is None:
            print(f"Warning: No target file found for {case_id}")
            continue
        
        target_files.append(target_file)
    
    if len(target_files) != len(pred_files):
        print(f"Warning: {len(pred_files)} prediction files but {len(target_files)} target files")
    
    # 평가 실행
    print(f"Evaluating {len(target_files)} cases...")
    results = evaluator.evaluate_dataset(pred_files, target_files, case_ids)
    
    # 결과 저장
    evaluator.save_results(results, output_path)
    
    # 주요 결과 출력
    print("\n" + "="*50)
    print("PANTHER Task1 Evaluation Results")
    print("="*50)
    
    for metric in evaluator.primary_metrics:
        mean_key = f'mean_{metric}'
        if mean_key in results:
            print(f"{metric}: {results[mean_key]:.4f} ± {results[f'std_{metric}']:.4f}")
    
    print(f"\nSuccessfully evaluated: {results['num_successful_cases']} cases")
    print(f"Failed cases: {results['num_failed_cases']}")
    print(f"Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PANTHER Task1 Evaluation')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing prediction files')
    parser.add_argument('--target_dir', type=str, required=True, help='Directory containing ground truth files')
    parser.add_argument('--output', type=str, default='panther_evaluation_results.json', help='Output file path')
    
    args = parser.parse_args()
    
    evaluate_panther_predictions(args.pred_dir, args.target_dir, args.output)