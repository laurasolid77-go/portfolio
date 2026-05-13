import sys
import os
from pathlib import Path
import pandas as pd
from rdkit import Chem

# 프로젝트 루트를 sys.path에 추가하여 src 모듈 임포트 보장
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.featurizer import featurize_dataframe

# QMugs orbital energies are stored in Hartree in the raw subset and converted to eV before training.
HARTREE_TO_EV = 27.2114

def canonicalize_smiles(smiles):
    if pd.isna(smiles) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None

def main():
    input_path = base_dir / "data" / "raw" / "qmugs_subset_matched_unique_50000.csv"
    output_path = base_dir / "data" / "processed" / "qmugs_features.csv"
    
    if not input_path.exists():
        print(f"에러: 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)
        
    print(f"데이터 로드 중: {input_path}")
    df = pd.read_csv(input_path)
    
    # 3. 필수 컬럼 확인
    required_cols = ['smiles', 'canonical_smiles', 'homo', 'lumo', 'gap']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"에러: 필수 컬럼이 누락되었습니다: {missing_cols}")
        sys.exit(1)
        
    num_original = len(df)
    
    # 5. 원본 Hartree target 보존
    df['homo_hartree'] = df['homo']
    df['lumo_hartree'] = df['lumo']
    df['gap_hartree'] = df['gap']
    
    # 6. eV 단위 target 컬럼 생성
    df['homo_ev'] = df['homo_hartree'] * HARTREE_TO_EV
    df['lumo_ev'] = df['lumo_hartree'] * HARTREE_TO_EV
    df['gap_ev'] = df['gap_hartree'] * HARTREE_TO_EV
    
    # 7. 기존 homo, lumo, gap 컬럼 제거
    df.drop(columns=['homo', 'lumo', 'gap'], inplace=True)
    
    # 8. RDKit canonical SMILES 재확인 및 정제
    print("SMILES 정규화 진행 중...")
    df['canonical_smiles_clean'] = df['canonical_smiles'].apply(canonicalize_smiles)
    
    # 9. invalid SMILES 제거
    df = df.dropna(subset=['canonical_smiles_clean']).copy()
    num_valid_smiles = len(df)
    
    # 기존 canonical_smiles 덮어쓰고 clean 컬럼 제거
    df['canonical_smiles'] = df['canonical_smiles_clean']
    df.drop(columns=['canonical_smiles_clean'], inplace=True)
    
    # 10. canonical_smiles 기준 중복 제거
    df = df.drop_duplicates(subset=['canonical_smiles'], keep='first').copy()
    num_dedup = len(df)
    
    # 11. target 결측치 제거
    df = df.dropna(subset=['homo_ev', 'lumo_ev', 'gap_ev']).copy()
    num_target_valid = len(df)
    
    # 12-13. src.featurizer를 사용하여 RDKit feature 새로 계산
    print("RDKit Feature 추출 중...")
    # 기존에 포함되어 있던 불필요한 descriptor 컬럼들은 featurizer에서 새 컬럼들과 섞일 수 있으므로
    # 기본/target 컬럼들만 분리하여 featurizer에 넘깁니다.
    core_cols = [
        'smiles', 'canonical_smiles', 
        'homo_ev', 'lumo_ev', 'gap_ev',
        'homo_hartree', 'lumo_hartree', 'gap_hartree'
    ]
    df_core = df[core_cols].copy()
    
    df_core = df_core.reset_index(drop=True)
    df_features_only = featurize_dataframe(df_core, smiles_col="canonical_smiles")
    
    # 15. 원본 컬럼과 Feature 병합 후 전부 NaN인 행 제거
    df_combined = pd.concat([df_core, df_features_only], axis=1)
    feature_cols = df_features_only.columns.tolist()
    df_combined = df_combined.dropna(subset=feature_cols, how='all').copy()
    num_feature_valid = len(df_combined)
    
    # 16. 최종 DataFrame 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    
    # 17. 통계 출력
    print("\n[전처리 완료 및 통계]")
    print(f"- 원본 QMugs subset 데이터 수       : {num_original}")
    print(f"- Invalid SMILES 제거 후 데이터 수  : {num_valid_smiles} (제거: {num_original - num_valid_smiles})")
    print(f"- 중복(canonical) 제거 후 데이터 수 : {num_dedup} (제거: {num_valid_smiles - num_dedup})")
    print(f"- Target 결측치 제거 후 데이터 수   : {num_target_valid} (제거: {num_dedup - num_target_valid})")
    print(f"- Feature 생성 실패 제거 후 데이터수: {num_feature_valid} (제거: {num_target_valid - num_feature_valid})")
    print(f"- 최종 저장 데이터 수               : {num_feature_valid}")
    print(f"- 생성된 Feature 컬럼 개수          : {len(feature_cols)}")
    
    if num_feature_valid > 0:
        print("\n[Target 요약 (eV 단위)]")
        print(f"HOMO_eV : min={df_combined['homo_ev'].min():.4f}, max={df_combined['homo_ev'].max():.4f}, mean={df_combined['homo_ev'].mean():.4f}")
        print(f"LUMO_eV : min={df_combined['lumo_ev'].min():.4f}, max={df_combined['lumo_ev'].max():.4f}, mean={df_combined['lumo_ev'].mean():.4f}")
        print(f"GAP_eV  : min={df_combined['gap_ev'].min():.4f}, max={df_combined['gap_ev'].max():.4f}, mean={df_combined['gap_ev'].mean():.4f}")
        
    print(f"\n[저장 경로] {output_path}")

if __name__ == "__main__":
    main()
