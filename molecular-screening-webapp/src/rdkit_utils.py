from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pandas as pd

def is_small_organic_molecule(mol):
    """
    유기 소분자 조건을 만족하는지 확인합니다.
    - 탄소(C) 원자 포함
    - 허용 원소: C, H, O, N, S, B, F, Cl, Br, I
    """
    if not mol:
        return False
        
    # 1. 탄소(C) 원자가 하나라도 있어야 유기 분자로 간주
    has_carbon = any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())
    if not has_carbon:
        return False
        
    # 2. 허용된 원소 목록
    allowed_elements = {'C', 'H', 'O', 'N', 'S', 'B', 'F', 'Cl', 'Br', 'I'}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in allowed_elements:
            return False
            
    return True

def calculate_descriptors(smiles):
    """SMILES로부터 RDKit 기술자를 계산합니다."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        
        # 유기 분자 필터링 (필요 시 호출부에서 사용하거나 여기서 통합 가능)
        # 여기서는 기본 기술자만 계산하여 반환합니다.
        
        stats = {}
        # 물리화학적 속성
        stats['mol_wt_rdkit'] = Descriptors.MolWt(mol)
        stats['heavy_atom_count'] = mol.GetNumHeavyAtoms()
        
        # 원자별 카운트 (H 제외)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        stats['C_count'] = atoms.count('C')
        stats['O_count'] = atoms.count('O')
        stats['N_count'] = atoms.count('N')
        stats['S_count'] = atoms.count('S')
        stats['halogen_count'] = atoms.count('F') + atoms.count('Cl') + atoms.count('Br') + atoms.count('I')
        
        # 고리 및 포화도
        stats['aromatic_ring_count'] = Descriptors.NumAromaticRings(mol)
        
        return stats
    except:
        return None

def add_rdkit_descriptors(df):
    """DataFrame의 SMILES 컬럼을 기반으로 RDKit 기술자 컬럼들을 추가합니다."""
    if "canonical_smiles" not in df.columns:
        return df
        
    results = []
    for smiles in df["canonical_smiles"]:
        desc = calculate_descriptors(smiles)
        results.append(desc if desc else {})
        
    desc_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)
