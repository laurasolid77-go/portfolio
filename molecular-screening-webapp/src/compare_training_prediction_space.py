import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

def add_missing_descriptors(df, smiles_col='canonical_smiles'):
    """데이터프레임에 누락된 RDKit descriptor를 계산하여 추가합니다."""
    # 계산할 항목이 있는지 확인
    needed_cols = [
        'mol_wt_rdkit', 'heavy_atom_count', 'total_atom_count', 
        'C_count', 'O_count', 'N_count', 'S_count', 
        'halogen_count', 'ring_count', 'aromatic_ring_count', 
        'B_count', 'P_count', 'Cl_count'
    ]
    
    missing = [c for c in needed_cols if c not in df.columns]
    if not missing:
        return df
        
    print(f"누락된 컬럼 계산 중: {missing}")
    
    # 미리 할당
    for c in missing:
        df[c] = 0
        
    halogens = {'F', 'Cl', 'Br', 'I'}
    
    for i, row in df.iterrows():
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
        if not mol:
            continue
            
        if 'mol_wt_rdkit' in missing: df.at[i, 'mol_wt_rdkit'] = Descriptors.MolWt(mol)
        if 'heavy_atom_count' in missing: df.at[i, 'heavy_atom_count'] = mol.GetNumHeavyAtoms()
        if 'total_atom_count' in missing: df.at[i, 'total_atom_count'] = mol.GetNumAtoms()
        if 'ring_count' in missing: df.at[i, 'ring_count'] = mol.GetRingInfo().NumRings()
        if 'aromatic_ring_count' in missing: df.at[i, 'aromatic_ring_count'] = Descriptors.NumAromaticRings(mol)
        
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if 'C_count' in missing: df.at[i, 'C_count'] = symbols.count('C')
        if 'O_count' in missing: df.at[i, 'O_count'] = symbols.count('O')
        if 'N_count' in missing: df.at[i, 'N_count'] = symbols.count('N')
        if 'S_count' in missing: df.at[i, 'S_count'] = symbols.count('S')
        if 'B_count' in missing: df.at[i, 'B_count'] = symbols.count('B')
        if 'P_count' in missing: df.at[i, 'P_count'] = symbols.count('P')
        if 'Cl_count' in missing: df.at[i, 'Cl_count'] = symbols.count('Cl')
        if 'halogen_count' in missing: df.at[i, 'halogen_count'] = sum(1 for s in symbols if s in halogens)
            
    return df

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qmugs_path = os.path.join(BASE_DIR, 'data', 'raw', 'qmugs_subset_50000.csv')
    library_path = os.path.join(BASE_DIR, 'data', 'processed', 'molecular_library.csv')
    reports_dir = os.path.join(BASE_DIR, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    print("[1] 데이터셋 로드...")
    df_qmugs = pd.read_csv(qmugs_path)
    df_lib = pd.read_csv(library_path)
    
    print(f"QMugs subset 크기: {len(df_qmugs)}")
    print(f"Molecular Library 크기: {len(df_lib)}")
    
    print("\n[2] 부족한 Descriptor 계산 (QMugs)...")
    # QMugs: total_atom_count, halogen_count 누락
    df_qmugs = add_missing_descriptors(df_qmugs, smiles_col='canonical_smiles')
    
    print("\n[3] 부족한 Descriptor 계산 (Molecular Library)...")
    # Library: B_count, P_count, Cl_count 등 누락 확인
    df_lib = add_missing_descriptors(df_lib, smiles_col='canonical_smiles')
    
    desc_list = [
        'mol_wt_rdkit', 'heavy_atom_count', 'total_atom_count', 
        'ring_count', 'aromatic_ring_count',
        'C_count', 'O_count', 'N_count', 'S_count', 
        'halogen_count', 'Cl_count', 'P_count', 'B_count'
    ]
    
    print("\n[4] 통계량 비교")
    stats_q = df_qmugs[desc_list].describe().T[['count', 'min', '25%', '50%', 'mean', '75%', 'max']]
    stats_l = df_lib[desc_list].describe().T[['count', 'min', '25%', '50%', 'mean', '75%', 'max']]
    
    # 두 통계를 합치기
    stats_q.columns = [f"QMugs_{c}" if c != 'count' else 'QMugs_count' for c in stats_q.columns]
    stats_l.columns = [f"Lib_{c}" if c != 'count' else 'Lib_count' for c in stats_l.columns]
    
    comparison = pd.concat([stats_q, stats_l], axis=1)
    
    # 보기 편하게 컬럼 순서 재배열
    col_order = []
    for c in ['count', 'min', '25%', '50%', 'mean', '75%', 'max']:
        col_order.extend([f"QMugs_{c}", f"Lib_{c}"])
    comparison = comparison[col_order]
    
    out_path = os.path.join(reports_dir, 'training_prediction_space_comparison.csv')
    comparison.to_csv(out_path)
    print(f"\n=> 결과 저장: {out_path}")
    
    print("\n========== Descriptor 별 요약표 ==========")
    for desc in desc_list:
        print(f"[{desc}]")
        print(f"  - Mean  : QMugs {stats_q.loc[desc, 'QMugs_mean']:.2f} | Lib {stats_l.loc[desc, 'Lib_mean']:.2f}")
        print(f"  - Median: QMugs {stats_q.loc[desc, 'QMugs_50%']:.2f} | Lib {stats_l.loc[desc, 'Lib_50%']:.2f}")
        print(f"  - Max   : QMugs {stats_q.loc[desc, 'QMugs_max']:.2f} | Lib {stats_l.loc[desc, 'Lib_max']:.2f}")
    
    print("\n========== 포함 비율 비교 ==========")
    for col, name in [('S_count', 'S'), ('halogen_count', 'Halogen'), ('Cl_count', 'Cl'), ('P_count', 'P'), ('B_count', 'B')]:
        q_ratio = (df_qmugs[col] > 0).mean() * 100
        l_ratio = (df_lib[col] > 0).mean() * 100
        print(f"- {name} 포함 비율: QMugs {q_ratio:.1f}% | Lib {l_ratio:.1f}%")
        
    print("\n========== Library 기준 추천 Subset 조건 ==========")
    lib_mw_95 = df_lib['mol_wt_rdkit'].quantile(0.95)
    lib_ha_95 = df_lib['heavy_atom_count'].quantile(0.95)
    lib_ta_95 = df_lib['total_atom_count'].quantile(0.95)
    lib_ar_95 = df_lib['aromatic_ring_count'].quantile(0.95)
    
    print(f"- MW <= {lib_mw_95:.1f} (Library 95th percentile)")
    print(f"- heavy_atom_count <= {int(lib_ha_95)} (Library 95th percentile)")
    print(f"- total_atom_count <= {int(lib_ta_95)} (Library 95th percentile)")
    print(f"- aromatic_ring_count <= {int(lib_ar_95)}")
    
    elements_in_lib = set()
    for col, ele in [('C_count','C'), ('O_count','O'), ('N_count','N'), ('S_count','S'), 
                     ('halogen_count','F/Cl/Br/I'), ('P_count','P'), ('B_count','B')]:
        if (df_lib[col] > 0).any():
            elements_in_lib.add(ele)
    print(f"- Allowed elements based on Library: {', '.join(sorted(elements_in_lib))}")
    print("- B_count > 0 은 out-of-domain으로 간주 (현재 Lib에 B가 없다면 제외)")
    print("- salt/mixture 제외")
    
    print("\n========== 최종 판단 ==========")
    is_too_drug_like = stats_q.loc['mol_wt_rdkit', 'QMugs_mean'] > stats_l.loc['mol_wt_rdkit', 'Lib_mean'] + 50
    if is_too_drug_like:
        print("=> 판단: QMugs subset이 Library에 비해 상당히 크거나 Drug-like 쪽으로 치우쳐 있습니다.")
    else:
        print("=> 판단: QMugs subset과 Library의 크기/복잡도 분포가 유사합니다.")
        
    print("=> QMugs 재구성 필요 여부: 분자량과 Heavy atom의 95% 구간이 크게 차이난다면, 제안된 95th percentile 기준으로 QMugs 데이터를 다시 필터링하여 재구성(Re-subset)하는 것이 적절합니다.")

if __name__ == "__main__":
    main()
