import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem

# 프로젝트 루트를 sys.path에 추가
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

def canonicalize_smiles(smiles):
    """
    SMILES 문자열을 RDKit canonical SMILES로 변환합니다.
    실패 시 None을 반환합니다.
    """
    if pd.isna(smiles) or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def main():
    # 1. 경로 설정
    qm9_path = BASE_DIR / "data" / "processed" / "qm9_features.csv"
    library_path = BASE_DIR / "data" / "processed" / "molecular_library.csv"
    
    output_checked_path = BASE_DIR / "data" / "processed" / "molecular_library_qm9_checked.csv"
    output_external_path = BASE_DIR / "data" / "processed" / "molecular_library_external_only.csv"
    
    # 2. 데이터 로드 (실행 시 에러 방지를 위한 존재 여부 확인)
    if not qm9_path.exists():
        print(f"Error: QM9 파일이 존재하지 않습니다. ({qm9_path})")
        sys.exit(1)
    if not library_path.exists():
        print(f"Error: Molecular Library 파일이 존재하지 않습니다. ({library_path})")
        sys.exit(1)
        
    print(f"Loading datasets...\n - QM9: {qm9_path}\n - Library: {library_path}")
    df_qm9 = pd.read_csv(qm9_path)
    df_lib = pd.read_csv(library_path)
    
    # 3. 필수 컬럼(canonical_smiles) 확인
    if "canonical_smiles" not in df_qm9.columns:
        print("Error: QM9 데이터에 'canonical_smiles' 컬럼이 없습니다.")
        sys.exit(1)
    if "canonical_smiles" not in df_lib.columns:
        print("Error: Molecular Library 데이터에 'canonical_smiles' 컬럼이 없습니다.")
        sys.exit(1)
        
    # 5 & 6. Canonicalize SMILES (정규화)
    print("\nCanonicalizing SMILES for comparison...")
    df_qm9["qm9_canonical_check"] = df_qm9["canonical_smiles"].apply(canonicalize_smiles)
    df_lib["library_canonical_check"] = df_lib["canonical_smiles"].apply(canonicalize_smiles)
    
    # 7. QM9 canonical SMILES set 생성 (결측치 제외)
    qm9_valid_smiles = df_qm9.dropna(subset=["qm9_canonical_check"])["qm9_canonical_check"].unique()
    qm9_smiles_set = set(qm9_valid_smiles)
    
    # 8. in_qm9 컬럼 추가
    df_lib["in_qm9"] = df_lib["library_canonical_check"].apply(
        lambda x: x in qm9_smiles_set if x is not None else False
    )
    
    # 9. Output 데이터 분리
    df_external = df_lib[df_lib["in_qm9"] == False].copy()
    
    # 저장
    df_lib.to_csv(output_checked_path, index=False)
    df_external.to_csv(output_external_path, index=False)
    
    # 10. 정보 출력
    qm9_total = len(df_qm9)
    qm9_valid = len(qm9_valid_smiles)
    lib_total = len(df_lib)
    lib_valid = df_lib["library_canonical_check"].notna().sum()
    overlap_count = df_lib["in_qm9"].sum()
    external_count = len(df_external)
    overlap_ratio = (overlap_count / lib_total) * 100 if lib_total > 0 else 0
    
    print("\n--- Summary ---")
    print(f"QM9 데이터 수: {qm9_total}")
    print(f"QM9 canonical valid 수: {qm9_valid}")
    print(f"Molecular Library 전체 데이터 수: {lib_total}")
    print(f"Molecular Library canonical valid 수: {lib_valid}")
    print(f"QM9과 중복된 분자 수: {overlap_count}")
    print(f"External only (예측 대상) 분자 수: {external_count}")
    print(f"중복 비율: {overlap_ratio:.2f}%")
    print(f"\nSaved outputs:")
    print(f" - Checked Library: {output_checked_path}")
    print(f" - External Only Library: {output_external_path}")
    
    # 11. 중복 분자 예시 출력
    if overlap_count > 0:
        print("\n--- 중복된 분자 예시 (최대 10개) ---")
        overlap_examples = df_lib[df_lib["in_qm9"]].head(10)
        
        # 출력할 때 필요한 컬럼들이 존재하는지 확인
        display_cols = []
        for col in ["cid", "iupac_name", "canonical_smiles", "library_canonical_check"]:
            if col in df_lib.columns:
                display_cols.append(col)
                
        # 보기 좋게 출력
        for _, row in overlap_examples.iterrows():
            info = []
            for col in display_cols:
                info.append(f"{col}: {row[col]}")
            print(" | ".join(info))

if __name__ == "__main__":
    main()
