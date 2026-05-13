import sys
import os

try:
    import datasets
except ImportError:
    print("datasets 라이브러리가 설치되어 있지 않습니다.")
    print("다음 명령어로 설치해주세요: pip install datasets")
    sys.exit(1)

from rdkit import Chem

def inspect_dataset():
    print("[1] Hugging Face datasets 라이브러리를 사용하여 PubChemQC B3LYP 데이터셋 연결 시도...")
    dataset_name = "molssiai-hub/pubchemqc-b3lyp"
    
    try:
        # streaming mode로 연결 (script 실행 허용을 위해 trust_remote_code=True 추가)
        ds = datasets.load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"데이터셋 연결에 실패했습니다: {e}")
        print("대안: OpenQDC나 다른 HF 데이터셋 이름이 필요할 수 있습니다.")
        return

    print("연결 성공! Streaming mode로 1000개 샘플을 순회하며 검사합니다.")
    
    sample_count = 0
    max_samples = 1000
    
    columns = None
    
    # 통계
    b_count = 0
    s_count = 0
    elements_found = set()
    
    # 컬럼 후보 추적
    homo_candidates = []
    lumo_candidates = []
    gap_candidates = []
    smiles_candidates = []
    cid_candidates = []

    for item in ds:
        if sample_count == 0:
            columns = list(item.keys())
            # 컬럼 이름 기반 후보 찾기
            homo_candidates = [c for c in columns if 'homo' in c.lower()]
            lumo_candidates = [c for c in columns if 'lumo' in c.lower()]
            gap_candidates = [c for c in columns if 'gap' in c.lower()]
            smiles_candidates = [c for c in columns if 'smile' in c.lower() or 'smi' in c.lower()]
            cid_candidates = [c for c in columns if 'cid' in c.lower() or 'id' in c.lower()]
            
        # RDKit 분석용 SMILES 추출
        smiles = None
        for sm_col in smiles_candidates:
            if item.get(sm_col):
                smiles = item[sm_col]
                break
                
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
                elements_found.update(symbols)
                if 'B' in symbols:
                    b_count += 1
                if 'S' in symbols:
                    s_count += 1
        
        sample_count += 1
        if sample_count >= max_samples:
            break

    print("\n[검사 결과]")
    print(f"- 조회한 샘플 수: {sample_count}")
    print(f"- 컬럼명 목록: {columns}")
    
    print("\n[주요 속성 컬럼 후보]")
    print(f"- CID 관련: {cid_candidates}")
    print(f"- SMILES 관련: {smiles_candidates}")
    print(f"- HOMO 관련: {homo_candidates}")
    print(f"- LUMO 관련: {lumo_candidates}")
    print(f"- Gap 관련: {gap_candidates}")
    
    print("\n[원소 조성 (RDKit 기준)]")
    print(f"- 발견된 전체 원소 종류: {sorted(list(elements_found))}")
    print(f"- B(Boron) 포함 분자 수: {b_count} / {sample_count} (약 {b_count/sample_count*100:.2f}%)")
    print(f"- S(Sulfur) 포함 분자 수: {s_count} / {sample_count} (약 {s_count/sample_count*100:.2f}%)")

    print("\n[결론 및 판단]")
    if homo_candidates and lumo_candidates and smiles_candidates:
        print("=> SMILES, HOMO, LUMO 정보를 모두 포함하고 있어 Training subset 구축이 **가능**할 것으로 판단됩니다.")
        print("=> Streaming 방식으로 50,000 ~ 100,000개를 순회하면서, 필요한 chemical space와 일치하는 조건(MW, 원소 등)을 필터링하여 다운로드 없이 메모리 상에서 subset을 구축할 수 있습니다.")
    else:
        print("=> 필수 속성 중 일부가 누락되어 있어 이 데이터셋만으로는 구축이 어려울 수 있습니다.")

if __name__ == "__main__":
    inspect_dataset()
