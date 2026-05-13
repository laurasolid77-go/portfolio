import sys
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# 프로젝트 루트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from src.featurizer import featurize_dataframe
except ImportError:
    from featurizer import featurize_dataframe

def find_input_file():
    """입력 파일을 정해진 우선순위에 따라 탐색합니다."""
    candidates = [
        BASE_DIR / "data" / "processed" / "molecular_library.csv",
        BASE_DIR / "data" / "raw" / "molecular_library.csv",
        BASE_DIR / "data" / "molecular_library.csv"
    ]
    
    for path in candidates:
        if path.exists():
            print(f"[1] 입력 파일 발견: {path}")
            return path
            
    raise FileNotFoundError("입력 파일(molecular_library.csv)을 찾을 수 없습니다. 경로를 확인해주세요.")

def find_smiles_column(df):
    """SMILES 정보가 담긴 컬럼을 자동으로 찾습니다."""
    candidates = ['smiles', 'SMILES', 'canonical_smiles', 'Canonical_SMILES', 'Canonical SMILES']
    for col in candidates:
        if col in df.columns:
            print(f"    - SMILES 컬럼 발견: '{col}'")
            return col
    raise KeyError("SMILES 컬럼을 찾을 수 없습니다. (후보: smiles, SMILES, canonical_smiles 등)")

def load_models():
    """저장된 최적 모델들을 불러옵니다."""
    models = {}
    targets = ['homo_ev', 'lumo_ev', 'gap_ev']
    
    print("\n[2] 모델 로드 중...")
    for target in targets:
        model_path = BASE_DIR / "models" / f"qmugs_{target}_best.pkl"
        if not model_path.exists():
            # 만약 best가 없으면 rf라도 시도 (사용자 요구사항은 best 우선)
            model_path = BASE_DIR / "models" / f"qmugs_{target}_rf.pkl"
            
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: qmugs_{target}_best.pkl 또는 qmugs_{target}_rf.pkl")
            
        try:
            with open(model_path, 'rb') as f:
                models[target] = pickle.load(f)
            print(f"    - {target} 모델 로드 완료: {model_path.name}")
        except ModuleNotFoundError as e:
            if 'xgboost' in str(e):
                print(f"\n[에러] XGBoost 모델을 로드하려면 xgboost 패키지가 필요합니다.")
                print("설치 명령: pip install xgboost lightgbm")
            raise e
            
    return models

def load_feature_columns():
    """학습 시 사용된 Feature 컬럼 리스트를 불러옵니다."""
    path = BASE_DIR / "models" / "qmugs_feature_columns.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Feature 리스트 파일을 찾을 수 없습니다: {path}")
        
    with open(path, 'rb') as f:
        features = pickle.load(f)
    print(f"    - Feature 리스트 로드 완료 ({len(features)}개)")
    return features

def predict_orbitals():
    """전체 예측 프로세스를 실행합니다."""
    # 1. 파일 및 환경 준비
    input_path = find_input_file()
    df = pd.read_csv(input_path)
    
    smiles_col = find_smiles_column(df)
    models = load_models()
    feature_cols = load_feature_columns()
    
    # 2. Descriptor 계산
    print("\n[3] 분자 Descriptor(Feature) 계산 중...")
    # featurize_dataframe은 invalid SMILES에 대해 빈 딕셔너리(결과적으로 NaN row)를 반환함
    feature_df = featurize_dataframe(df, smiles_col=smiles_col)
    
    # 학습 시 사용한 컬럼 순서와 일치시키기
    # missing column이 있으면 에러 발생 (학습/예측 featurizer 불일치 방지)
    try:
        X = feature_df[feature_cols]
    except KeyError as e:
        missing = set(feature_cols) - set(feature_df.columns)
        raise KeyError(f"Descriptor 계산 결과에 필수 Feature가 누락되었습니다: {missing}")

    # 3. Invalid SMILES 체크
    invalid_mask = X.isnull().any(axis=1)
    invalid_indices = df.index[invalid_mask].tolist()
    valid_count = len(df) - len(invalid_indices)
    
    if len(invalid_indices) > 0:
        print(f"    - 경고: {len(invalid_indices)}개의 Invalid SMILES가 발견되었습니다. (예측 건너뜀)")
        if len(invalid_indices) <= 5:
            print(f"    - 해당 인덱스: {invalid_indices}")
        else:
            print(f"    - 처음 5개 인덱스: {invalid_indices[:5]}")

    # 4. 예측 수행
    print("\n[4] 궤도 에너지(HOMO, LUMO, Gap) 예측 중...")
    results_df = df.copy()
    
    # Valid한 데이터에 대해서만 예측 수행
    X_valid = X[~invalid_mask]
    
    if not X_valid.empty:
        for target, model in models.items():
            pred_col = f"pred_{target}"
            # 예측 수행 및 소수점 4자리 반올림
            preds = model.predict(X_valid)
            
            # 전체 DataFrame에 매핑 (invalid는 NaN 유지)
            results_df.loc[~invalid_mask, pred_col] = np.round(preds, 4)
    else:
        print("    - 예측할 수 있는 유효한 분자가 없습니다.")
        for target in models.keys():
            results_df[f"pred_{target}"] = np.nan

    # 5. 결과 저장
    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "molecular_library_with_orbitals.csv"
    
    results_df.to_csv(output_path, index=False)
    
    # 6. 요약 정보 출력
    print_prediction_summary(input_path, output_path, len(df), valid_count, len(invalid_indices), results_df)

def print_prediction_summary(input_path, output_path, total, valid, invalid, df):
    """최종 실행 결과를 요약하여 출력합니다."""
    print("\n" + "="*60)
    print("분자 궤도 에너지 예측 완료 요약")
    print("="*60)
    print(f"- 입력 파일: {input_path}")
    print(f"- 전체 분자 수: {total}")
    print(f"- 유효 분자 수: {valid}")
    print(f"- 무효 분자 수: {invalid}")
    print(f"- 결과 저장 경로: {output_path}")
    
    print("\n[예측값 통계 (eV)]")
    pred_cols = ['pred_homo_ev', 'pred_lumo_ev', 'pred_gap_ev']
    stats = df[pred_cols].agg(['min', 'max', 'mean', 'std']).T
    print(stats.to_string())
    print("="*60)

def main():
    try:
        predict_orbitals()
        print(f"\n[성공] 모든 예측 과정이 완료되었습니다.")
    except Exception as e:
        print(f"\n[에러] 예측 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
