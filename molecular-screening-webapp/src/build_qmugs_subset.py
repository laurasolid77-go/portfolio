import sys
import os
import pandas as pd
import numpy as np

# 데이터셋 연결을 위한 라이브러리
try:
    import datasets
except ImportError:
    print("datasets 라이브러리가 필요합니다. pip install datasets")
    sys.exit(1)

from rdkit import Chem
from rdkit.Chem import Descriptors

def main():
    print("[1] Hugging Face에서 QMugs_Summary 데이터셋 스트리밍 시작...")
    
    try:
        # JuIm/QMugs_Summary는 QMugs의 주요 성질과 SMILES가 정리된 버전입니다.
        ds = datasets.load_dataset('JuIm/QMugs_Summary', split='train', streaming=True)
    except Exception as e:
        print(f"데이터셋 접근 실패: {e}")
        return

    print("\n[2] Unique Molecule 기준 Matched subset 수집 중... (목표: 50,000개)")
    
    target_count = 50000
    collected_data = []
    seen_canonical_smiles = set()
    
    # 허용 원소: C, H, N, O, F, S, Cl (B, P 등은 명시적 제외)
    allowed_elements = {'C', 'H', 'N', 'O', 'F', 'S', 'Cl'}
    
    # 통계용 변수
    b_count = 0
    p_count = 0
    s_count = 0
    cl_count = 0
    skip_duplicate_conformer = 0
    
    for row in ds:
        if len(collected_data) >= target_count:
            break
            
        smiles = row.get("smiles")
        if not smiles: continue
        
        # HOMO, LUMO, GAP 확보 (DFT 우선, 없으면 GFN2)
        homo = row.get("DFT_HOMO_ENERGY") if row.get("DFT_HOMO_ENERGY") is not None else row.get("GFN2_HOMO_ENERGY")
        lumo = row.get("DFT_LUMO_ENERGY") if row.get("DFT_LUMO_ENERGY") is not None else row.get("GFN2_LUMO_ENERGY")
        
        if homo is None or lumo is None:
            continue
            
        gap = row.get("DFT_HOMO_LUMO_GAP") if row.get("DFT_HOMO_LUMO_GAP") is not None else row.get("GFN2_HOMO_LUMO_GAP")
        if gap is None:
            gap = lumo - homo
            
        # RDKit Mol 생성 가능 여부 확인
        mol = Chem.MolFromSmiles(str(smiles))
        if not mol: continue
        
        # Canonical SMILES 생성하여 중복(conformer) 체크
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        if not canonical_smiles: continue
        
        if canonical_smiles in seen_canonical_smiles:
            skip_duplicate_conformer += 1
            continue
            
        # Salt/mixture 제외
        if len(Chem.GetMolFrags(mol)) > 1:
            continue
            
        # 원소 확인 (B, P 포함 여부 필터링)
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if any(sym not in allowed_elements for sym in symbols):
            continue
            
        # 속성 계산 및 필터링
        mw = Descriptors.MolWt(mol)
        ha_count = mol.GetNumHeavyAtoms()
        total_atoms = mol.GetNumAtoms()
        aro_rings = Descriptors.NumAromaticRings(mol)
        
        # Matched 조건 적용 (molecular_library 분포에 맞춤)
        if mw > 480 or ha_count > 33 or aro_rings > 4:
            continue
            
        rings = mol.GetRingInfo().NumRings()
        
        c_c = symbols.count('C')
        n_c = symbols.count('N')
        o_c = symbols.count('O')
        f_c = symbols.count('F')
        s_c = symbols.count('S')
        cl_c = symbols.count('Cl')
        b_c = symbols.count('B')
        p_c = symbols.count('P')
        
        # 통계 카운트 (필터링 통과 후)
        if b_c > 0: b_count += 1
        if p_c > 0: p_count += 1
        if s_c > 0: s_count += 1
        if cl_c > 0: cl_count += 1
        
        # 중복이 아닌 신규 분자인 경우 수집
        seen_canonical_smiles.add(canonical_smiles)
        collected_data.append({
            "smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "homo": homo,
            "lumo": lumo,
            "gap": gap,
            "mol_wt_rdkit": mw,
            "heavy_atom_count": ha_count,
            "total_atom_count": total_atoms,
            "ring_count": rings,
            "aromatic_ring_count": aro_rings,
            "C_count": c_c,
            "N_count": n_c,
            "O_count": o_c,
            "F_count": f_c,
            "S_count": s_c,
            "Cl_count": cl_c,
            "B_count": b_c,
            "P_count": p_c
        })
        
        if len(collected_data) % 5000 == 0:
            print(f"  -> 수집 진행률 (Unique Molecule): {len(collected_data)} / {target_count} (Skip conformers: {skip_duplicate_conformer})")

    print("\n[3] 수집 완료! CSV 저장 진행...")
    
    df = pd.DataFrame(collected_data)
    
    # 저장 경로 설정
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(BASE_DIR, "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "qmugs_subset_matched_unique_50000.csv")
    
    df.to_csv(out_path, index=False)
    
    # 중복 체크 최종 검증
    duplicate_count = df['canonical_smiles'].duplicated().sum()
    unique_count = df['canonical_smiles'].nunique()
    
    print("\n[수집 및 검증 결과 요약]")
    print(f"- 최종 저장 row 수: {len(df)}")
    print(f"- Unique canonical_smiles 수: {unique_count}")
    print(f"- 중복 canonical_smiles 수: {duplicate_count} (0이어야 함)")
    print(f"- Skip한 중복 Conformer row 수: {skip_duplicate_conformer}")
    print(f"- B 포함 분자 수: {b_count} (0이어야 함)")
    print(f"- P 포함 분자 수: {p_count} (0이어야 함)")
    print(f"- S 포함 분자 수: {s_count} ({s_count/len(df)*100:.1f}%)")
    print(f"- Cl 포함 분자 수: {cl_count} ({cl_count/len(df)*100:.1f}%)")
    
    if len(df) > 0:
        print(f"\n- MW: Min {df['mol_wt_rdkit'].min():.1f}, Max {df['mol_wt_rdkit'].max():.1f}, Mean {df['mol_wt_rdkit'].mean():.1f}, Median {df['mol_wt_rdkit'].median():.1f}")
        print(f"- Heavy Atom: Min {df['heavy_atom_count'].min()}, Max {df['heavy_atom_count'].max()}, Mean {df['heavy_atom_count'].mean():.1f}, Median {df['heavy_atom_count'].median():.1f}")
        print(f"- Total Atom: Min {df['total_atom_count'].min()}, Max {df['total_atom_count'].max()}, Mean {df['total_atom_count'].mean():.1f}, Median {df['total_atom_count'].median():.1f}")
        print(f"- Ring: Min {df['ring_count'].min()}, Max {df['ring_count'].max()}, Mean {df['ring_count'].mean():.1f}, Median {df['ring_count'].median():.1f}")
        print(f"- Aromatic Ring: Min {df['aromatic_ring_count'].min()}, Max {df['aromatic_ring_count'].max()}, Mean {df['aromatic_ring_count'].mean():.1f}, Median {df['aromatic_ring_count'].median():.1f}")
        print(f"- HOMO: Min {df['homo'].min():.4f}, Max {df['homo'].max():.4f}, Mean {df['homo'].mean():.4f}")
        print(f"- LUMO: Min {df['lumo'].min():.4f}, Max {df['lumo'].max():.4f}, Mean {df['lumo'].mean():.4f}")
        print(f"- Gap: Min {df['gap'].min():.4f}, Max {df['gap'].max():.4f}, Mean {df['gap'].mean():.4f}")
        
    print(f"\n- 저장 경로: {out_path}")
    
    # 50,000개 수집 실패 시 보고
    if len(df) < target_count:
        print(f"\n[주의] 목표치 {target_count}개에 도달하지 못했습니다. 수집된 개수: {len(df)}")
    
    print("\n[4] Molecular Library와 주요 Descriptor 평균 비교")
    lib_path = os.path.join(BASE_DIR, 'data', 'processed', 'molecular_library.csv')
    
    if os.path.exists(lib_path):
        df_lib = pd.read_csv(lib_path)
        
        # Helper functions for comparison
        def get_halogens(row):
            if 'halogen_count' in row: return row['halogen_count']
            s = str(row.get('canonical_smiles', ''))
            return s.count('F') + s.count('Cl') + s.count('Br') + s.count('I')
            
        def get_s(row):
            if 'S_count' in row: return row['S_count']
            return str(row.get('canonical_smiles', '')).count('S') + str(row.get('canonical_smiles', '')).count('s')
            
        def get_cl(row):
            if 'Cl_count' in row: return row['Cl_count']
            return str(row.get('canonical_smiles', '')).count('Cl')

        lib_mw_mean = df_lib['mol_wt_rdkit'].mean() if 'mol_wt_rdkit' in df_lib.columns else np.nan
        lib_ha_mean = df_lib['heavy_atom_count'].mean() if 'heavy_atom_count' in df_lib.columns else np.nan
        lib_total_mean = df_lib['total_atom_count'].mean() if 'total_atom_count' in df_lib.columns else np.nan
        lib_ring_mean = df_lib['ring_count'].mean() if 'ring_count' in df_lib.columns else np.nan
        lib_ar_mean = df_lib['aromatic_ring_count'].mean() if 'aromatic_ring_count' in df_lib.columns else np.nan
        
        lib_s_ratio = (df_lib.apply(get_s, axis=1) > 0).mean() * 100
        lib_hal_ratio = (df_lib.apply(get_halogens, axis=1) > 0).mean() * 100
        lib_cl_ratio = (df_lib.apply(get_cl, axis=1) > 0).mean() * 100
        
        new_hal_ratio = ((df['F_count'] + df['Cl_count']) > 0).mean() * 100
        new_s_ratio = (df['S_count'] > 0).mean() * 100
        new_cl_ratio = (df['Cl_count'] > 0).mean() * 100
        
        print("\n=== 속성 평균/비율 비교 (Library vs New Matched Unique Subset) ===")
        print(f"1. MW 평균:")
        print(f"   - Library           : {lib_mw_mean:.1f}")
        print(f"   - Matched Unique    : {df['mol_wt_rdkit'].mean():.1f}")
        
        print(f"\n2. Heavy Atom 평균:")
        print(f"   - Library           : {lib_ha_mean:.1f}")
        print(f"   - Matched Unique    : {df['heavy_atom_count'].mean():.1f}")
        
        print(f"\n3. Total Atom 평균:")
        print(f"   - Library           : {lib_total_mean:.1f}")
        print(f"   - Matched Unique    : {df['total_atom_count'].mean():.1f}")
        
        print(f"\n4. Ring / Aromatic Ring 평균:")
        print(f"   - Library           : {lib_ring_mean:.1f} / {lib_ar_mean:.1f}")
        print(f"   - Matched Unique    : {df['ring_count'].mean():.1f} / {df['aromatic_ring_count'].mean():.1f}")
        
        print(f"\n5. S / Halogen / Cl 포함 비율:")
        print(f"   - Library           : S {lib_s_ratio:.1f}% / Hal {lib_hal_ratio:.1f}% / Cl {lib_cl_ratio:.1f}%")
        print(f"   - Matched Unique    : S {new_s_ratio:.1f}% / Hal {new_hal_ratio:.1f}% / Cl {new_cl_ratio:.1f}%")

if __name__ == "__main__":
    main()
