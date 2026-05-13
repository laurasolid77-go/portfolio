import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem

def canonicalize_smiles(smiles):
    if pd.isna(smiles) or not smiles:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    return None

def main():
    # 1. pathlib.Path를 사용한 경로 설정
    base_dir = Path(__file__).resolve().parent.parent
    qmugs_path = base_dir / "data" / "raw" / "qmugs_subset_matched_50000.csv"
    library_path = base_dir / "data" / "processed" / "molecular_library.csv"
    out_checked_path = base_dir / "data" / "processed" / "molecular_library_qmugs_checked.csv"
    out_external_path = base_dir / "data" / "processed" / "molecular_library_qmugs_external_only.csv"
    
    # 2. pandas 로드
    if not qmugs_path.exists() or not library_path.exists():
        print(f"파일을 찾을 수 없습니다.\nQMugs: {qmugs_path}\nLibrary: {library_path}")
        sys.exit(1)
        
    df_qmugs = pd.read_csv(qmugs_path)
    df_lib = pd.read_csv(library_path)
    
    # 3. canonical_smiles 존재 여부 확인
    if 'canonical_smiles' not in df_qmugs.columns:
        print("에러: qmugs_subset_matched_50000.csv에 canonical_smiles 컬럼이 없습니다.")
        sys.exit(1)
    if 'canonical_smiles' not in df_lib.columns:
        print("에러: molecular_library.csv에 canonical_smiles 컬럼이 없습니다.")
        sys.exit(1)
        
    print("[1] 데이터 로드 완료 및 SMILES 정규화 진행 중...")
    
    # 5. QMugs SMILES 정규화
    df_qmugs['qmugs_canonical_check'] = df_qmugs['canonical_smiles'].apply(canonicalize_smiles)
    
    # 6. Library SMILES 정규화
    df_lib['library_canonical_check'] = df_lib['canonical_smiles'].apply(canonicalize_smiles)
    
    # 7. QMugs canonical SMILES set 생성
    qmugs_smiles_set = set(df_qmugs['qmugs_canonical_check'].dropna())
    
    # 8. in_qmugs 컬럼 추가
    df_lib['in_qmugs'] = df_lib['library_canonical_check'].isin(qmugs_smiles_set)
    
    # 9. Boron 포함 여부 확인 (B_count)
    if 'B_count' not in df_lib.columns:
        print("molecular_library에 B_count가 없어 RDKit으로 계산합니다.")
        def count_boron(smiles):
            mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
            if mol:
                return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'B')
            return 0
        df_lib['B_count'] = df_lib['canonical_smiles'].apply(count_boron)
    
    # 10. 파일 저장
    df_lib.to_csv(out_checked_path, index=False)
    
    df_external = df_lib[~df_lib['in_qmugs']].copy()
    df_external.to_csv(out_external_path, index=False)
    
    # 11. 통계 출력
    num_qmugs = len(df_qmugs)
    num_qmugs_valid = len(qmugs_smiles_set)
    num_lib = len(df_lib)
    num_lib_valid = df_lib['library_canonical_check'].notna().sum()
    num_overlap = df_lib['in_qmugs'].sum()
    num_external = len(df_external)
    overlap_ratio = (num_overlap / num_lib) * 100 if num_lib > 0 else 0
    num_b_containing = (df_lib['B_count'] > 0).sum()
    
    print("\n[2] Overlap Check 통계 결과")
    print(f"- matched QMugs subset 데이터 수       : {num_qmugs}")
    print(f"- matched QMugs canonical valid 수     : {num_qmugs_valid}")
    print(f"- molecular_library 전체 데이터 수     : {num_lib}")
    print(f"- molecular_library canonical valid 수 : {num_lib_valid}")
    print(f"- QMugs와 중복된 분자 수 (Overlap)     : {num_overlap}")
    print(f"- External-only 분자 수                : {num_external}")
    print(f"- 중복 비율 (%)                        : {overlap_ratio:.2f}%")
    print(f"- Library 내 B_count > 0 분자 수       : {num_b_containing}")
    
    print(f"\n[저장 경로]")
    print(f"1) 전체 결과 : {out_checked_path}")
    print(f"2) 외부 예측 : {out_external_path}")
    
    # 12. 중복된 분자 예시 출력
    if num_overlap > 0:
        print("\n[QMugs와 중복된 분자 예시 (최대 10개)]")
        overlap_samples = df_lib[df_lib['in_qmugs']].head(10)
        # 만약 cid, iupac_name 등 컬럼이 없으면 에러가 날 수 있으므로 확인 후 출력
        cols_to_print = ['cid', 'iupac_name', 'canonical_smiles', 'library_canonical_check']
        cols_to_print = [c for c in cols_to_print if c in overlap_samples.columns]
        print(overlap_samples[cols_to_print].to_string(index=False))
    
    # 13. B_count > 0인 분자 예시 출력
    if num_b_containing > 0:
        print("\n[B (Boron) 포함 분자 예시 (최대 10개)]")
        b_samples = df_lib[df_lib['B_count'] > 0].head(10)
        cols_to_print = ['cid', 'iupac_name', 'canonical_smiles', 'B_count']
        cols_to_print = [c for c in cols_to_print if c in b_samples.columns]
        print(b_samples[cols_to_print].to_string(index=False))
        
    # 14. 필수 컬럼 유지 여부 검증
    required_cols = [
        'cid', 'iupac_name', 'canonical_smiles', 'mol_wt_rdkit', 'heavy_atom_count',
        'total_atom_count', 'ring_count', 'aromatic_ring_count',
        'C_count', 'O_count', 'N_count', 'S_count', 'halogen_count'
    ]
    missing_cols = [c for c in required_cols if c not in df_external.columns]
    
    print("\n[3] 필수 컬럼 검증")
    if missing_cols:
        print(f"경고: 다음 필수 컬럼이 external_only CSV에 없습니다: {missing_cols}")
    else:
        print("모든 필수 컬럼이 external_only CSV에 잘 유지되어 있습니다.")

if __name__ == "__main__":
    main()
