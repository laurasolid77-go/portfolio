import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def smiles_to_features(smiles):
    """
    SMILES 문자열을 입력받아 HOMO, LUMO, Gap 예측에 필요한 RDKit 특징량(Features)을 추출합니다.
    
    Args:
        smiles (str): 분자의 SMILES 문자열
        
    Returns:
        dict: 수치형 특징량을 담은 딕셔너리. 실패 시 None 반환.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 특징량을 저장할 딕셔너리
        features = {}
        
        # 1. RDKit 주요 Descriptors 계산
        features["MolWt"] = Descriptors.MolWt(mol)
        features["HeavyAtomCount"] = Descriptors.HeavyAtomCount(mol)
        features["NumHAcceptors"] = Descriptors.NumHAcceptors(mol)
        features["NumHDonors"] = Descriptors.NumHDonors(mol)
        features["TPSA"] = Descriptors.TPSA(mol)
        features["MolLogP"] = Descriptors.MolLogP(mol)
        features["NumRotatableBonds"] = Descriptors.NumRotatableBonds(mol)
        features["FractionCSP3"] = Descriptors.FractionCSP3(mol)
        features["BertzCT"] = Descriptors.BertzCT(mol)
        features["LabuteASA"] = Descriptors.LabuteASA(mol)
        features["BalabanJ"] = Descriptors.BalabanJ(mol)
        
        # 고리 관련 특징 (rdMolDescriptors 활용)
        features["RingCount"] = rdMolDescriptors.CalcNumRings(mol)
        features["NumAromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        
        # 2. 원소별 Count 계산 (수소 포함)
        # AddHs를 사용하여 명시적인 수소를 추가한 Mol 객체 생성
        mol_with_hs = Chem.AddHs(mol)
        
        # 원소 카운트 초기화
        element_counts = {
            "C_count": 0, "H_count": 0, "N_count": 0, "O_count": 0,
            "F_count": 0, "S_count": 0, "Cl_count": 0, "Br_count": 0, "I_count": 0
        }
        
        for atom in mol_with_hs.GetAtoms():
            symbol = atom.GetSymbol()
            key = f"{symbol}_count"
            if key in element_counts:
                element_counts[key] += 1
                
        features.update(element_counts)
        
        return features

    except Exception as e:
        # 조용히 None 반환 (로그가 필요하면 추가 가능)
        return None

def featurize_dataframe(df, smiles_col="canonical_smiles"):
    """
    DataFrame의 SMILES 컬럼을 기준으로 모든 분자의 특징량을 계산하여 새로운 DataFrame을 반환합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        smiles_col (str): SMILES 정보가 담긴 컬럼명
        
    Returns:
        pd.DataFrame: 계산된 특징량들로 구성된 데이터프레임 (원본 인덱스 유지)
    """
    feature_list = []
    
    for smiles in df[smiles_col]:
        feats = smiles_to_features(smiles)
        if feats is None:
            # RDKit 로드 실패 시 NaN으로 채운 딕셔너리 생성 (첫 번째 성공 케이스의 키 기준)
            feature_list.append({}) 
        else:
            feature_list.append(feats)
            
    # 리스트를 데이터프레임으로 변환
    feature_df = pd.DataFrame(feature_list, index=df.index)
    
    return feature_df

# 테스트 코드
if __name__ == "__main__":
    test_smiles = ["CCO", "c1ccccc1", "invalid_smiles"]
    
    print("--- Individual SMILES Test ---")
    for s in test_smiles:
        print(f"\nSMILES: {s}")
        res = smiles_to_features(s)
        if res:
            for k, v in res.items():
                print(f"  {k}: {v}")
        else:
            print("  Failed to extract features.")
            
    print("\n--- DataFrame Test ---")
    df_test = pd.DataFrame({"canonical_smiles": test_smiles})
    features_df = featurize_dataframe(df_test)
    print(features_df)
