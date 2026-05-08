import sys
import os
import pandas as pd
import random
import time
from rdkit import Chem

# 프로젝트 루트를 경로에 추가하여 src 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pubchem_utils import fetch_properties_by_cids
from src.rdkit_utils import is_small_organic_molecule, calculate_descriptors

def build_organic_library(target_size=5000, batch_size=200):
    """
    무작위 CID 샘플링을 통해 유기 소분자 라이브러리를 구축합니다.
    - 대상: PubChem CID 1 ~ 120,000,000
    - 필터: 유기 분자 조건(C 포함, 금속 제외), MW(50-600), Heavy Atoms(3-80)
    """
    random.seed(42)
    valid_molecules = []
    total_raw_fetched = 0
    batch_count = 0
    
    print(f"[Start] 유기 소분자 라이브러리 구축 시작 (목표: {target_size}개)", flush=True)
    print(f"   [Condition: MW 50-600, Heavy Atoms 3-80, C-containing organic only]", flush=True)
    
    # 중복 방지를 위한 집합
    seen_cids = set()

    while len(valid_molecules) < target_size:
        batch_count += 1
        
        # 1~100만 사이의 무작위 CID 생성 (중복 제외)
        current_batch_cids = []
        while len(current_batch_cids) < batch_size:
            rcid = random.randint(1, 1000000)
            if rcid not in seen_cids:
                current_batch_cids.append(rcid)
                seen_cids.add(rcid)
        
        # 1. PubChem API 데이터 가져오기
        raw_df = fetch_properties_by_cids(current_batch_cids)
        if raw_df is None or raw_df.empty:
            # 배치 전체가 유효하지 않은 경우(드문 경우)
            continue
            
        total_raw_fetched += len(raw_df)
        
        # 2. 유기 분자 조건 및 물리화학적 조건 필터링
        for _, row in raw_df.iterrows():
            if len(valid_molecules) >= target_size:
                break
                
            # CanonicalSMILES 또는 ConnectivitySMILES 중 있는 것을 사용
            smiles = row.get("CanonicalSMILES") or row.get("ConnectivitySMILES")
            if not isinstance(smiles, str): continue
            
            # RDKit Mol 객체 생성
            mol = Chem.MolFromSmiles(smiles)
            if not mol: continue
            
            # [필터 1] 유기 분자 판별 (C 포함, 허용 원소 내 존재)
            if not is_small_organic_molecule(mol): continue
            
            # [필터 2] 물리화학적 속성 체크
            mw = float(row.get("MolecularWeight", 0))
            heavy_atoms = mol.GetNumHeavyAtoms()
            
            if not (50 <= mw <= 600): continue
            if not (3 <= heavy_atoms <= 80): continue
            
            # [성공] 모든 조건을 통과하면 기술자 계산 및 저장
            desc = calculate_descriptors(smiles)
            if desc:
                molecule_data = {
                    "cid": int(row["CID"]),
                    "iupac_name": row.get("IUPACName", "N/A"),
                    "canonical_smiles": smiles,
                    **desc
                }
                valid_molecules.append(molecule_data)
        
        # 진행 상황 출력
        if batch_count % 1 == 0:
            print(f"Batch {batch_count:03d} | Total Raw: {total_raw_fetched:5d} | Valid Organic: {len(valid_molecules):4d}/{target_size}", flush=True)
        
        # API 부하 방지를 위한 짧은 대기
        time.sleep(0.2)
        
        # 무한 루프 방지 (안전 장치)
        if batch_count > 2000:
            print("[Warning] 최대 시도 횟수에 도달하여 수집을 조기 종료합니다.", flush=True)
            break

    # 3. 데이터 저장
    if valid_molecules:
        final_df = pd.DataFrame(valid_molecules)
        
        # 디렉토리 생성
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        
        raw_path = "data/raw/pubchem_raw.csv"
        proc_path = "data/processed/molecular_library.csv"
        
        final_df.to_csv(raw_path, index=False)
        final_df.to_csv(proc_path, index=False)
        
        print(f"\n[Success] 라이브러리 구축 완료!", flush=True)
        print(f"   - Saved at: {proc_path}", flush=True)
        print(f"   - Final Size: {len(final_df)} Molecules", flush=True)
    else:
        print("\n[Error] 조건을 만족하는 분자를 찾지 못했습니다.", flush=True)

if __name__ == "__main__":
    build_organic_library()
