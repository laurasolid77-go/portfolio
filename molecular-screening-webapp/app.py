import streamlit as st
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from io import BytesIO

# 페이지 설정
st.set_page_config(page_title="Molecular Screening Prototype", layout="wide")

# 작용기별 SMARTS 패턴 정의
FUNCTIONAL_GROUPS = {
    "C=O (Carbonyl)": "[CX3]=[OX1]",
    "-COOH (Carboxylic Acid)": "[CX3](=O)[OX2H1]",
    "-COOR (Ester)": "[CX3](=O)[OX2][#6]",
    "-CONH- / -CONR- (Amide)": "[CX3](=O)[NX3]",
    "-OH (Hydroxyl)": "[OX2H]",
    "-NH2 / -NHR / -NR2 (Amine)": "[NX3;!$(NC=O)]",
    "-CN (Nitrile)": "[CX2]#N",
    "-NO2 (Nitro)": "[NX3](=O)=O"
}

# 세션 상태 초기화
if "filters_applied" not in st.session_state:
    st.session_state["filters_applied"] = False
if "current_page" not in st.session_state:
    st.session_state["current_page"] = 1
if "items_per_page" not in st.session_state:
    st.session_state["items_per_page"] = 10

def calculate_total_atoms(smiles):
    """SMILES로부터 수소를 포함한 전체 원자 수를 계산합니다."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol_with_hs = Chem.AddHs(mol)
            return mol_with_hs.GetNumAtoms()
    except:
        pass
    return None

def calculate_rings(smiles):
    """SMILES로부터 전체 고리(Ring) 수를 계산합니다."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return rdMolDescriptors.CalcNumRings(mol)
    except:
        pass
    return 0

def has_selected_functionalities(smiles, selected_patterns):
    """SMILES가 선택된 작용기 중 하나라도 포함하는지 확인합니다(OR 조건)."""
    if not selected_patterns:
        return True
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False
        for pattern in selected_patterns:
            query = Chem.MolFromSmarts(pattern)
            if query and mol.HasSubstructMatch(query):
                return True
        return False
    except:
        return False

def load_data(file_path):
    """라이브러리 데이터를 로드하고 전처리합니다."""
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        update_needed = False
        if "total_atom_count" not in df.columns:
            with st.spinner("Atom Count 계산 중..."):
                df["total_atom_count"] = df["canonical_smiles"].apply(calculate_total_atoms)
            update_needed = True
        if "ring_count" not in df.columns:
            with st.spinner("Ring Count 계산 중..."):
                df["ring_count"] = df["canonical_smiles"].apply(calculate_rings)
            update_needed = True
        if update_needed:
            df = df.dropna(subset=["total_atom_count", "ring_count"])
            df["total_atom_count"] = df["total_atom_count"].astype(int)
            df["ring_count"] = df["ring_count"].astype(int)
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

@st.cache_data
def convert_df_to_csv(df):
    """DataFrame을 CSV로 변환합니다."""
    return df.to_csv(index=False).encode('utf-8-sig')

def render_molecule_card(row):
    """분자 정보를 매우 컴팩트한 카드 형태로 렌더링합니다."""
    with st.container(border=True):
        c1, c2 = st.columns([1, 2.2])
        with c1:
            mol = Chem.MolFromSmiles(row["canonical_smiles"])
            if mol:
                img = Draw.MolToImage(mol, size=(180, 180))
                st.image(img, use_container_width=True)
            else:
                st.write("No Image")
        with c2:
            name = row['iupac_name'] if pd.notna(row['iupac_name']) else 'Unnamed'
            display_name = f"{name[:35]}..." if len(str(name)) > 35 else name
            st.markdown(f"**{display_name}**")
            st.markdown(f"<small>CID: {int(row['cid'])}</small>", unsafe_allow_html=True)
            smiles = row["canonical_smiles"]
            display_smiles = smiles[:45] + "..." if len(smiles) > 45 else smiles
            st.code(display_smiles, language="text")
            st.markdown(
                f"<div style='line-height:1.2;'><small>"
                f"⚖️ MW: <b>{row['mol_wt_rdkit']:.1f}</b> | "
                f"💎 Heavy: <b>{int(row['heavy_atom_count'])}</b> | "
                f"⚛️ Total: <b>{int(row['total_atom_count'])}</b><br>"
                f"🏗️ Ring: <b>{int(row['ring_count'])}</b> (Aro: {int(row['aromatic_ring_count'])})<br>"
                f"🧪 C/O/N/S: <b>{int(row['C_count'])}/{int(row['O_count'])}/{int(row['N_count'])}/{int(row['S_count'])}</b> | "
                f"🛡️ Hal: <b>{int(row['halogen_count'])}</b>"
                f"</small></div>", 
                unsafe_allow_html=True
            )

# --- 동기화 필터 관련 함수 ---

def init_filter_state(df, col_name):
    abs_min, abs_max = float(df[col_name].min()), float(df[col_name].max())
    if f"{col_name}_min" not in st.session_state: st.session_state[f"{col_name}_min"] = abs_min
    if f"{col_name}_max" not in st.session_state: st.session_state[f"{col_name}_max"] = abs_max
    if f"{col_name}_slider" not in st.session_state: st.session_state[f"{col_name}_slider"] = (abs_min, abs_max)

def slider_callback(col_name):
    val = st.session_state[f"{col_name}_slider"]
    st.session_state[f"{col_name}_min"], st.session_state[f"{col_name}_max"] = val[0], val[1]

def input_callback(col_name):
    st.session_state[f"{col_name}_slider"] = (st.session_state[f"{col_name}_min"], st.session_state[f"{col_name}_max"])

def get_filter_values_synced(df, col_name, label):
    abs_min, abs_max = float(df[col_name].min()), float(df[col_name].max())
    if abs_min == abs_max: return None, None, None
    init_filter_state(df, col_name)
    st.sidebar.markdown(f"**{label}**")
    st.sidebar.slider(f"{label} Slider", abs_min, abs_max, key=f"{col_name}_slider", on_change=slider_callback, args=(col_name,), label_visibility="collapsed")
    c1, c2 = st.sidebar.columns(2)
    with c1: st.number_input("Min", key=f"{col_name}_min", on_change=input_callback, args=(col_name,))
    with c2: st.number_input("Max", key=f"{col_name}_max", on_change=input_callback, args=(col_name,))
    return st.session_state[f"{col_name}_min"], st.session_state[f"{col_name}_max"], (f"'{label}'의 Min이 Max보다 큽니다." if st.session_state[f"{col_name}_min"] > st.session_state[f"{col_name}_max"] else None)

# --- 메인 앱 로직 ---

def main():
    st.title("🔬 Molecular Screening Prototype")
    lib_path = "data/processed/molecular_library.csv"
    df = load_data(lib_path)
    if df is None:
        st.error(f"데이터 파일 부재: `{lib_path}`")
        return
    st.info(f"현재 라이브러리 규모: 총 **{len(df)}**개 분자")

    # 1. 사이드바 필터링
    st.sidebar.header("🔍 1st Stage Filters")
    all_filters, filter_errors = {}, []
    filter_configs = [
        ("📐 Basic Size", [("mol_wt_rdkit", "Molecular Weight"), ("heavy_atom_count", "Heavy Atom Count"), ("total_atom_count", "Total Atom Count")]),
        ("⚛️ Atom Count", [("C_count", "Carbon Count (C)"), ("O_count", "Oxygen Count (O)"), ("N_count", "Nitrogen Count (N)"), ("S_count", "Sulfur Count (S)"), ("halogen_count", "Halogen Count")]),
        ("🏗️ Structure", [("ring_count", "Ring Count"), ("aromatic_ring_count", "Aromatic Ring Count")])
    ]
    for section, configs in filter_configs:
        st.sidebar.subheader(section)
        for col, lab in configs:
            vmin, vmax, err = get_filter_values_synced(df, col, lab)
            if vmin is not None: all_filters[col] = (vmin, vmax)
            if err: filter_errors.append(err)

    if st.sidebar.button("Apply Screening Filters", type="primary"):
        st.session_state["filters_applied"] = True
        st.session_state["current_page"] = 1

    # 2. 결과 표시
    if st.session_state["filters_applied"]:
        if filter_errors:
            for err in filter_errors: st.error(err)
            return
        primary_filtered_df = df.copy()
        for col, (vmin, vmax) in all_filters.items():
            primary_filtered_df = primary_filtered_df[(primary_filtered_df[col] >= vmin) & (primary_filtered_df[col] <= vmax)]
        st.divider()
        st.subheader("🧪 2nd Stage: Functionality Filters")
        selected_patterns = []
        fn_cols = st.columns(4)
        for i, (name, pattern) in enumerate(FUNCTIONAL_GROUPS.items()):
            with fn_cols[i % 4]:
                if st.checkbox(name, key=f"cb_{name}"): selected_patterns.append(pattern)
        if selected_patterns:
            primary_filtered_df["fn_match"] = primary_filtered_df["canonical_smiles"].apply(lambda x: has_selected_functionalities(x, selected_patterns))
            final_filtered_df = primary_filtered_df[primary_filtered_df["fn_match"]].copy()
        else:
            final_filtered_df = primary_filtered_df

        final_mols = len(final_filtered_df)
        if final_mols > 0:
            st.divider()
            # 1. 상단 메시지 박스
            st.success(f"✅ **필터링 완료:** 총 {final_mols}개 분자가 선택되었습니다.")
            
            # 2. 다운로드 툴바 영역
            t_col1, t_col2 = st.columns([2, 1])
            with t_col1:
                st.markdown(
                    "<p style='color: grey; font-size: 0.9rem; padding-top: 10px;'>"
                    "현재 필터링 결과를 CSV로 저장할 수 있습니다."
                    "</p>", 
                    unsafe_allow_html=True
                )
            with t_col2:
                csv_data = convert_df_to_csv(final_filtered_df.drop(columns=["fn_match"]) if "fn_match" in final_filtered_df.columns else final_filtered_df)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv_data,
                    file_name="filtered_molecules.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_btn_toolbar"
                )
            
            # --- 카드 렌더링 ---
            items_per_page = st.session_state["items_per_page"]
            total_pages = (final_mols - 1) // items_per_page + 1
            st.session_state["current_page"] = min(st.session_state["current_page"], total_pages)
            start_idx = (st.session_state["current_page"] - 1) * items_per_page
            page_data = final_filtered_df.iloc[start_idx : start_idx + items_per_page]
            for i in range(0, len(page_data), 2):
                row_cols = st.columns(2)
                with row_cols[0]: render_molecule_card(page_data.iloc[i])
                if i + 1 < len(page_data):
                    with row_cols[1]: render_molecule_card(page_data.iloc[i+1])
            
            # --- 하단 페이지네이션 UI ---
            st.write("")
            st.divider()
            _, c_pagination, _ = st.columns([1, 4, 1])
            with c_pagination:
                nav_cols = st.columns([1.2, 0.8, 1.5, 0.8])
                with nav_cols[0]:
                    st.session_state["items_per_page"] = st.selectbox("Items per page", [10, 20, 30], index=[10, 20, 30].index(st.session_state["items_per_page"]), label_visibility="collapsed")
                    st.caption("Page size")
                with nav_cols[1]:
                    st.write("")
                    if st.button("⬅️ Prev", disabled=(st.session_state["current_page"] <= 1), use_container_width=True, key="btn_prev"):
                        st.session_state["current_page"] -= 1
                        st.rerun()
                with nav_cols[2]:
                    st.markdown(f"<div style='text-align: center; padding-top: 10px;'><small>Page <b>{st.session_state['current_page']}</b> / {total_pages}</small></div>", unsafe_allow_html=True)
                with nav_cols[3]:
                    st.write("")
                    if st.button("Next ➡️", disabled=(st.session_state["current_page"] >= total_pages), use_container_width=True, key="btn_next"):
                        st.session_state["current_page"] += 1
                        st.rerun()
        else:
            st.warning("⚠️ 조건을 만족하는 분자가 없습니다.")
    else:
        st.write("👈 사이드바에서 조건을 설정하고 **Apply Screening Filters**를 클릭하세요.")

if __name__ == "__main__":
    main()
