import sys
import os
import time

try:
    import datasets
except ImportError:
    print("datasets 라이브러리가 설치되어 있지 않습니다.")
    sys.exit(1)

from rdkit import Chem
from rdkit.Chem import Descriptors

def check_conditions(mol, allowed_elements, max_mw, max_heavy):
    """조건 A와 B 필터링을 위한 함수"""
    if not mol:
        return False
        
    # Salt/mixture 제외 (단일 분자인지 확인)
    if len(Chem.GetMolFrags(mol)) > 1:
        return False
        
    # 허용 원소 확인
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_elements:
            return False
            
    # 분자량 및 heavy atom 확인
    mw = Descriptors.MolWt(mol)
    ha_count = mol.GetNumHeavyAtoms()
    
    if mw > max_mw or ha_count > max_heavy:
        return False
        
    return True

def main():
    print("[1] Hugging Face에서 QMugs_Summary 데이터셋 초기화 시도...")
    
    try:
        ds = datasets.load_dataset('JuIm/QMugs_Summary', split='train', streaming=True)
    except Exception as e:
        print(f"데이터셋 접근 실패: {e}")
        return

    print("\n[2] 데이터셋 샘플 로드 및 속성 검사...")
    
    max_samples = 5000
    sample_count = 0
    
    b_count = 0
    s_count = 0
    elements_found = set()
    mw_list = []
    ha_list = []
    ring_list = []
    aromatic_list = []
    
    allowed_elements = {'C', 'H', 'B', 'N', 'O', 'F', 'P', 'S', 'Cl'}
    cond_a_pass = 0
    cond_b_pass = 0
    
    keys_found = None
    
    for row in ds:
        if sample_count == 0:
            keys_found = list(row.keys())
            print(f"발견된 Data 속성/키: {keys_found}")
            
        smiles = row.get("smiles")
        if smiles:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol:
                symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
                elements_found.update(symbols)
                
                if 'B' in symbols: b_count += 1
                if 'S' in symbols: s_count += 1
                
                mw = Descriptors.MolWt(mol)
                ha = mol.GetNumHeavyAtoms()
                rings = mol.GetRingInfo().NumRings()
                aro_rings = Descriptors.NumAromaticRings(mol)
                
                mw_list.append(mw)
                ha_list.append(ha)
                ring_list.append(rings)
                aromatic_list.append(aro_rings)
                
                if check_conditions(mol, allowed_elements, 500, 40):
                    cond_a_pass += 1
                if check_conditions(mol, allowed_elements, 400, 30):
                    cond_b_pass += 1
                    
        sample_count += 1
        if sample_count >= max_samples:
            break

    print(f"\n[통계 결과 - 샘플 {sample_count}개 기준]")
    print(f"- 포함 원소 목록: {sorted(list(elements_found))}")
    print(f"- B 포함 분자 수: {b_count} ({b_count/sample_count*100:.2f}%)")
    print(f"- S 포함 분자 수: {s_count} ({s_count/sample_count*100:.2f}%)")
    
    if mw_list:
        print(f"- MW 분포: Min {min(mw_list):.1f}, Max {max(mw_list):.1f}, Mean {sum(mw_list)/len(mw_list):.1f}")
        print(f"- Heavy Atom 분포: Min {min(ha_list)}, Max {max(ha_list)}, Mean {sum(ha_list)/len(ha_list):.1f}")
        print(f"- Ring 분포: Min {min(ring_list)}, Max {max(ring_list)}, Mean {sum(ring_list)/len(ring_list):.1f}")
        print(f"- Aromatic Ring 분포: Min {min(aromatic_list)}, Max {max(aromatic_list)}, Mean {sum(aromatic_list)/len(aromatic_list):.1f}")
        
    print(f"\n[Subset 조건 통과 비율]")
    print(f"- 조건 A (MW<=500, HA<=40, 특정원소): {cond_a_pass}개 통과 ({(cond_a_pass/sample_count)*100:.2f}%)")
    print(f"- 조건 B (MW<=400, HA<=30, 특정원소): {cond_b_pass}개 통과 ({(cond_b_pass/sample_count)*100:.2f}%)")
    
    print("\n[답변 요약]")
    print("- QMugs에서 SMILES를 쉽게 얻을 수 있는가? " + ("예" if 'smiles' in keys_found else "아니오/추가확인필요"))
    print("- HOMO/LUMO를 쉽게 얻을 수 있는가? " + ("예" if any('homo' in k.lower() for k in keys_found) else "아니오/추가확인필요"))
    print("- gap은 직접 제공되는가, 아니면 계산해야 하는가? " + ("직접 제공됨" if any('gap' in k.lower() for k in keys_found) else "계산해야 함"))

if __name__ == "__main__":
    main()
