#!/usr/bin/env python3
"""
PANTHER Challenge Local Inference Test Script
Docker 컨테이너 빌드 전 로컬에서 테스트
"""

import os
import sys
import argparse
from pathlib import Path
import SimpleITK as sitk
import numpy as np

# 현재 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_submission import PANTHERInference


def test_single_case(model_path: str, config_path: str, test_image: str, output_dir: str):
    """단일 케이스 테스트"""
    
    print("🧪 Testing PANTHER Inference...")
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print(f"Test image: {test_image}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Inference 객체 생성
    inferencer = PANTHERInference(
        model_path=model_path,
        config_path=config_path,
        device='cuda'
    )
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 테스트 이미지 처리
    test_file = Path(test_image)
    output_file = output_path / f"{test_file.stem}_prediction.nii.gz"
    
    print(f"🔄 Processing {test_file.name}...")
    
    # 원본 이미지 정보 출력
    original_image = sitk.ReadImage(test_image)
    original_array = sitk.GetArrayFromImage(original_image)
    
    print(f"📊 Original image info:")
    print(f"  - Shape: {original_array.shape}")
    print(f"  - Spacing: {original_image.GetSpacing()}")
    print(f"  - Origin: {original_image.GetOrigin()}")
    print(f"  - Size: {original_image.GetSize()}")
    print()
    
    # 예측 수행
    inferencer.predict_single_case(
        input_path=test_image,
        output_path=str(output_file)
    )
    
    # 결과 검증
    print(f"📊 Prediction result:")
    if output_file.exists():
        prediction_image = sitk.ReadImage(str(output_file))
        prediction_array = sitk.GetArrayFromImage(prediction_image)
        
        print(f"  - Output file: {output_file}")
        print(f"  - Shape: {prediction_array.shape}")
        print(f"  - Spacing: {prediction_image.GetSpacing()}")
        print(f"  - Origin: {prediction_image.GetOrigin()}")
        print(f"  - Size: {prediction_image.GetSize()}")
        print(f"  - Unique values: {np.unique(prediction_array)}")
        
        # 형태 일치 확인
        if prediction_array.shape == original_array.shape:
            print("  - ✅ Shape matches original!")
        else:
            print(f"  - ❌ Shape mismatch! Original: {original_array.shape}, Prediction: {prediction_array.shape}")
        
        # Spacing 일치 확인
        original_spacing = original_image.GetSpacing()
        prediction_spacing = prediction_image.GetSpacing()
        spacing_diff = [abs(o - p) for o, p in zip(original_spacing, prediction_spacing)]
        
        if all(diff < 1e-6 for diff in spacing_diff):
            print("  - ✅ Spacing matches original!")
        else:
            print(f"  - ❌ Spacing mismatch! Original: {original_spacing}, Prediction: {prediction_spacing}")
    
    print("\n🎉 Test completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Test PANTHER Inference locally")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model checkpoint (auto-detect if not provided)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (auto-detect if not provided)")
    parser.add_argument("--test_image", type=str, required=True,
                       help="Path to test image (.mha or .nii.gz)")
    parser.add_argument("--output_dir", type=str, default="./test_output",
                       help="Output directory for predictions")
    
    args = parser.parse_args()
    
    # 모델과 설정 파일 자동 감지
    if not args.model or not args.config:
        print("🔍 Auto-detecting model and config files...")
        experiments_dir = Path("./experiments")
        if experiments_dir.exists():
            # SegResNet 실험 폴더 찾기
            segresnet_dirs = list(experiments_dir.glob("segresnet_*"))
            if segresnet_dirs:
                latest_dir = sorted(segresnet_dirs)[-1]  # 가장 최신
                if not args.model:
                    args.model = str(latest_dir / "best_checkpoint.pth")
                if not args.config:
                    args.config = str(latest_dir / "config.yaml")
                print(f"📁 Using experiment: {latest_dir.name}")
    
    # 파일 존재 확인
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.exists(args.test_image):
        raise FileNotFoundError(f"Test image not found: {args.test_image}")
    
    print(f"📦 Model: {args.model}")
    print(f"⚙️ Config: {args.config}")
    print(f"🖼️ Test image: {args.test_image}")
    print()
    
    # 테스트 실행
    test_single_case(
        model_path=args.model,
        config_path=args.config,
        test_image=args.test_image,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()