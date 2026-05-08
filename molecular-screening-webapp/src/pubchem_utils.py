import requests
import pandas as pd
import io
import time

def fetch_pubchem_data(keyword, max_records=500):
    """(기존 방식) 키워드를 기반으로 PubChem 데이터를 검색하여 가져옵니다."""
    # 하위 호환성을 위해 유지하거나 필요 없으면 삭제 가능합니다.
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{keyword}/property/CanonicalSMILES,IUPACName,MolecularWeight/CSV"
    try:
        response = requests.get(url, params={"name_type": "word"}, timeout=30)
        if response.status_code == 200:
            return pd.read_csv(io.StringIO(response.text))
        return None
    except Exception as e:
        print(f"Error fetching data for {keyword}: {e}")
        return None

def fetch_properties_by_cids(cid_list):
    """CID 목록을 기반으로 PubChem에서 대량의 속성 정보를 가져옵니다."""
    if not cid_list:
        return None
        
    cids_str = ",".join(map(str, cid_list))
    # PUG-REST API 사용
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids_str}/property/CanonicalSMILES,IUPACName,MolecularWeight/CSV"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return pd.read_csv(io.StringIO(response.text))
        elif response.status_code == 404:
            # 해당 CID들이 존재하지 않는 경우
            return None
        else:
            print(f"API Error (Status {response.status_code}) for CID batch.")
            return None
    except Exception as e:
        print(f"Network error during CID batch fetch: {e}")
        return None
