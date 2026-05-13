import sys
import os
import io
from pathlib import Path
import pandas as pd
import requests

def get_column_mapping(columns):
    """주어진 컬럼 목록에서 요구하는 4가지 물성에 해당하는 실제 컬럼명을 찾습니다."""
    mapping = {}
    
    # SMILES 후보
    smiles_cands = ["smiles", "mol_id", "SMILES"]
    for cand in smiles_cands:
        if cand in columns:
            mapping["smiles"] = cand
            break
            
    # HOMO 후보
    homo_cands = ["homo", "HOMO"]
    for cand in homo_cands:
        if cand in columns:
            mapping["homo"] = cand
            break
            
    # LUMO 후보
    lumo_cands = ["lumo", "LUMO"]
    for cand in lumo_cands:
        if cand in columns:
            mapping["lumo"] = cand
            break
            
    # GAP 후보
    gap_cands = ["gap", "GAP", "homo_lumo_gap", "HOMO_LUMO_gap"]
    for cand in gap_cands:
        if cand in columns:
            mapping["gap"] = cand
            break
            
    return mapping

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUT_PATH = BASE_DIR / "data" / "raw" / "qm9.csv"
    URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
    
    # data/raw 폴더 생성
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading QM9 dataset from {URL} ...")
    
    try:
        # 우선 pandas로 직접 읽기 시도
        df_raw = pd.read_csv(URL)
        print("pandas.read_csv를 통한 다운로드 성공!")
    except Exception as e:
        print(f"pandas 직접 다운로드 실패: {e}")
        print("Fallback: requests 모듈을 사용하여 다운로드 시도 중...")
        try:
            response = requests.get(URL)
            response.raise_for_status()
            csv_content = io.StringIO(response.text)
            df_raw = pd.read_csv(csv_content)
            print("requests 모듈을 통한 다운로드 성공!")
        except Exception as e2:
            print(f"requests를 통한 다운로드도 실패했습니다: {e2}")
            sys.exit(1)
            
    print("\n--- 다운로드된 원본 CSV 컬럼 목록 ---")
    print(list(df_raw.columns))
    
    print("\n--- 원본 CSV 앞 5행 Preview ---")
    print(df_raw.head())
    
    # 컬럼 매핑 찾기
    mapping = get_column_mapping(df_raw.columns)
    
    # 필수 4개 컬럼이 모두 찾아졌는지 확인
    required = ["smiles", "homo", "lumo", "gap"]
    if not all(k in mapping for k in required):
        print("\nCould not find required QM9 columns. Please check printed column names.")
        print(f"찾은 매핑 결과: {mapping}")
        sys.exit(1)
        
    print(f"\n매핑 성공: {mapping}")
    
    # 추출 및 이름 변경
    df_extracted = df_raw[[mapping["smiles"], mapping["homo"], mapping["lumo"], mapping["gap"]]].copy()
    df_extracted.columns = ["smiles", "homo", "lumo", "gap"]
    
    # 저장
    df_extracted.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n--- 최종 저장 결과 ---")
    print(f"저장 경로: {OUTPUT_PATH}")
    print(f"총 Row 수: {len(df_extracted)}")
    print(f"컬럼 목록: {list(df_extracted.columns)}")
    
    print("\n[앞 5행 Preview]")
    print(df_extracted.head())
    
    print("\n[결측치 개수]")
    print(df_extracted.isnull().sum())
    
    print("\n[물성 Min / Max / Mean]")
    for col in ["homo", "lumo", "gap"]:
        val_min = df_extracted[col].min()
        val_max = df_extracted[col].max()
        val_mean = df_extracted[col].mean()
        print(f"- {col}: Min = {val_min:.4f}, Max = {val_max:.4f}, Mean = {val_mean:.4f}")

if __name__ == "__main__":
    main()
