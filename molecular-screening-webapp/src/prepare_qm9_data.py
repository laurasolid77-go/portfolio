import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem

# 프로젝트 루트를 sys.path에 추가하여 src 모듈을 안정적으로 import
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.featurizer import featurize_dataframe

HARTREE_TO_EV = 27.2114

def canonicalize_smiles(smiles):
    """
    SMILES 문자열을 RDKit canonical SMILES로 변환합니다.
    실패 시 None을 반환합니다.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def main():
    # 경로 설정
    raw_data_path = BASE_DIR / "data" / "raw" / "qm9.csv"
    output_data_path = BASE_DIR / "data" / "processed" / "qm9_features.csv"
    
    if not raw_data_path.exists():
        print(f"Error: {raw_data_path} 파일이 존재하지 않습니다.")
        sys.exit(1)
        
    print(f"Loading QM9 raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    
    # 필수 컬럼 확인
    required_columns = ["smiles", "homo", "lumo", "gap"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: 입력 데이터에 다음 필수 컬럼이 없습니다: {missing_columns}")
        sys.exit(1)
        
    initial_count = len(df)
    print(f"원본 QM9 데이터 수: {initial_count}")
    
    # DeepChem QM9 target values are in Hartree-like atomic units in this CSV, so they are converted to eV before training.
    print(f"Converting Hartree to eV (1 Hartree = {HARTREE_TO_EV} eV)...")
    df["homo_hartree"] = df["homo"]
    df["lumo_hartree"] = df["lumo"]
    df["gap_hartree"] = df["gap"]
    
    df["homo_ev"] = df["homo_hartree"] * HARTREE_TO_EV
    df["lumo_ev"] = df["lumo_hartree"] * HARTREE_TO_EV
    df["gap_ev"] = df["gap_hartree"] * HARTREE_TO_EV
    
    # Canonical SMILES 변환
    print("Canonical SMILES로 변환 중...")
    df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
    
    # invalid SMILES 제거
    df_valid = df.dropna(subset=["canonical_smiles"]).copy()
    valid_count = len(df_valid)
    print(f"invalid SMILES 제거 후 데이터 수: {valid_count} (제거됨: {initial_count - valid_count})")
    
    # 중복 제거
    df_unique = df_valid.drop_duplicates(subset=["canonical_smiles"], keep="first").copy()
    unique_count = len(df_unique)
    print(f"canonical_smiles 중복 제거 후 데이터 수: {unique_count} (제거됨: {valid_count - unique_count})")
    
    # Target 결측치 제거 (eV 단위 기준)
    df_target_valid = df_unique.dropna(subset=["homo_ev", "lumo_ev", "gap_ev"]).copy()
    target_valid_count = len(df_target_valid)
    print(f"target 결측치 제거 후 데이터 수: {target_valid_count} (제거됨: {unique_count - target_valid_count})")
    
    # RDKit Feature 추출
    print("RDKit feature 추출 중 (이 작업은 시간이 걸릴 수 있습니다)...")
    features_df = featurize_dataframe(df_target_valid, smiles_col="canonical_smiles")
    
    # DataFrame 병합 (기존 homo, lumo, gap 제외)
    final_cols = [
        "smiles", "canonical_smiles", 
        "homo_ev", "lumo_ev", "gap_ev",
        "homo_hartree", "lumo_hartree", "gap_hartree"
    ]
    df_final = pd.concat([df_target_valid[final_cols], features_df], axis=1)
    
    # Feature가 전부 NaN인 행 제거
    feature_columns = features_df.columns
    df_final = df_final.dropna(subset=feature_columns, how="all").copy()
    final_count = len(df_final)
    
    print("\n--- Summary ---")
    print(f"최종 저장 데이터 수: {final_count} (Feature 추출 실패로 제거됨: {target_valid_count - final_count})")
    print(f"Feature column 개수: {len(feature_columns)}")
    
    print("\n[물성 Min / Max / Mean (eV 단위)]")
    for col in ["homo_ev", "lumo_ev", "gap_ev"]:
        val_min = df_final[col].min()
        val_max = df_final[col].max()
        val_mean = df_final[col].mean()
        print(f"- {col}: Min = {val_min:.4f}, Max = {val_max:.4f}, Mean = {val_mean:.4f}")
    
    # 최종 결과 저장
    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_data_path, index=False)
    print(f"\n최종 데이터 저장 완료! Output path: {output_data_path}")

if __name__ == "__main__":
    main()
