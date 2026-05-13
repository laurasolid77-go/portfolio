import sys
import os
import pandas as pd
import random
import time
from rdkit import Chem

# 프로젝트 루트를 경로에 추가
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(BASE_DIR))

from src.pubchem_utils import fetch_properties_by_cids
from src.rdkit_utils import is_small_organic_molecule, calculate_descriptors

def get_canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None

def main():
    target_total = 5000
    batch_size = 200

    # 1. 기존 데이터 로드
    qm9_path = os.path.join(BASE_DIR, "data/processed/qm9_features.csv")
    lib_path = os.path.join(BASE_DIR, "data/processed/molecular_library.csv")
    ext_path = os.path.join(BASE_DIR, "data/processed/molecular_library_external_only.csv")

    qm9_df = pd.read_csv(qm9_path)
    lib_df = pd.read_csv(lib_path)
    ext_df = pd.read_csv(ext_path)

    current_ext_count = len(ext_df)
    n_needed = target_total - current_ext_count

    print(f"기존 external-only 개수: {current_ext_count}")
    print(f"부족한 개수: {n_needed}")

    if n_needed <= 0:
        print("이미 5000개 이상의 external 분자가 있습니다. 작업을 종료합니다.")
        return

    # 2. 중복 검사용 집합 구성
    qm9_smiles_set = set(qm9_df['canonical_smiles'].dropna())
    
    # molecular_library 생성 시 사용했던 SMILES와 library_canonical_check를 포함하여 중복 방지
    lib_smiles_set = set()
    if 'canonical_smiles' in lib_df.columns:
        lib_smiles_set.update(lib_df['canonical_smiles'].dropna())
    
    ext_smiles_set = set()
    if 'canonical_smiles' in ext_df.columns:
        ext_smiles_set.update(ext_df['canonical_smiles'].dropna())
    if 'library_canonical_check' in ext_df.columns:
        ext_smiles_set.update(ext_df['library_canonical_check'].dropna())

    # 3. 추가 수집
    random.seed(42)
    new_molecules = []
    seen_cids = set(lib_df['cid']) if 'cid' in lib_df.columns else set()
    
    # stats
    try_count = 0
    total_fetched = 0
    overlap_qm9 = 0
    overlap_lib = 0
    overlap_ext = 0
    overlap_internal = 0
    
    internal_smiles_set = set()

    print(f"\nPubChem에서 {n_needed}개의 새로운 분자 수집을 시작합니다...")

    while len(new_molecules) < n_needed:
        try_count += 1
        
        current_batch_cids = []
        while len(current_batch_cids) < batch_size:
            rcid = random.randint(1, 1000000)
            if rcid not in seen_cids:
                current_batch_cids.append(rcid)
                seen_cids.add(rcid)
        
        raw_df = fetch_properties_by_cids(current_batch_cids)
        if raw_df is None or raw_df.empty:
            continue
            
        total_fetched += len(raw_df)
        
        for _, row in raw_df.iterrows():
            if len(new_molecules) >= n_needed:
                break
                
            raw_smiles = row.get("CanonicalSMILES") or row.get("ConnectivitySMILES")
            if not isinstance(raw_smiles, str): continue
            
            mol = Chem.MolFromSmiles(raw_smiles)
            if not mol: continue
            
            # 필터 1: 유기 소분자
            if not is_small_organic_molecule(mol): continue
            
            # 필터 2: 속성
            mw = float(row.get("MolecularWeight", 0))
            heavy_atoms = mol.GetNumHeavyAtoms()
            if not (50 <= mw <= 600): continue
            if not (3 <= heavy_atoms <= 80): continue
            
            # Canonical SMILES 변환
            canon_smiles = Chem.MolToSmiles(mol, canonical=True)
            if not canon_smiles: continue

            # 중복 검사
            if canon_smiles in qm9_smiles_set:
                overlap_qm9 += 1
                continue
            if canon_smiles in lib_smiles_set:
                overlap_lib += 1
                continue
            if canon_smiles in ext_smiles_set:
                overlap_ext += 1
                continue
            if canon_smiles in internal_smiles_set:
                overlap_internal += 1
                continue
                
            # 유효 분자 통과
            desc = calculate_descriptors(raw_smiles)
            if desc:
                molecule_data = {
                    "cid": int(row["CID"]),
                    "iupac_name": row.get("IUPACName", "N/A"),
                    "canonical_smiles": raw_smiles,
                    "library_canonical_check": canon_smiles,
                    "in_qm9": False,
                    **desc
                }
                new_molecules.append(molecule_data)
                internal_smiles_set.add(canon_smiles)

        print(f"Batch {try_count:03d} | Fetched: {total_fetched:5d} | New Valid: {len(new_molecules):4d}/{n_needed}", flush=True)
        time.sleep(0.2)
        
        if try_count > 1000:
            print("[Warning] 최대 시도 횟수에 도달하여 조기 종료합니다.")
            break

    # 4. 저장
    new_df = pd.DataFrame(new_molecules)
    # 기존 데이터프레임과 합치기 (컬럼 순서는 기존과 동일하게 유지)
    final_df = pd.concat([ext_df, new_df], ignore_index=True)
    
    # 기존 컬럼만 유지하도록 순서 조정
    expected_cols = ext_df.columns.tolist()
    # 혹시 누락된 컬럼이 있으면 채워넣음 (보통은 calculate_descriptors가 동일하게 생성함)
    for col in expected_cols:
        if col not in final_df.columns:
            final_df[col] = None
    final_df = final_df[expected_cols]

    out_path = os.path.join(BASE_DIR, "data/processed/molecular_library_external_5000.csv")
    final_df.to_csv(out_path, index=False)

    print("\n[수집 완료 및 요약 보고]")
    print(f"- 기존 external-only 개수: {current_ext_count}")
    print(f"- 부족한 개수: {n_needed}")
    print(f"- 추가 수집 시도 횟수: {try_count}")
    print(f"- 추가 후보 총 수집 수: {total_fetched}")
    print(f"- QM9 중복으로 제거된 수: {overlap_qm9}")
    print(f"- 기존 molecular_library 중복으로 제거된 수: {overlap_lib}")
    print(f"- external_only 기존 분자 중복으로 제거된 수: {overlap_ext}")
    print(f"- 추가 수집분 내부 중복 제거 수: {overlap_internal}")
    print(f"- 최종 신규 확보 분자 수: {len(new_molecules)}")
    print(f"- 최종 external-only 총 개수: {len(final_df)}")
    print(f"- 저장 경로: {out_path}")

if __name__ == "__main__":
    main()
