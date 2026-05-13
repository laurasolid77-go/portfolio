import streamlit as st
import pandas as pd
import os
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors
from io import BytesIO
import streamlit.components.v1 as components

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent

def get_data_path():
    """데이터 파일의 우선순위에 따라 경로를 반환합니다."""
    candidates = [
        BASE_DIR / "data" / "processed" / "molecular_library_with_orbitals.csv",
        BASE_DIR / "data" / "processed" / "molecular_library.csv",
        BASE_DIR / "data" / "raw" / "molecular_library.csv"
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

DATA_PATH = get_data_path()

# 과학적 색상 팔레트 설정
SCIENTIFIC_COLORS = {
    "HOMO": "#2C5F8A",  # Deep blue
    "LUMO": "#2A9D8F",  # Teal
    "GAP": "#E9A03B",   # Muted orange
    "Gray": "#4A5568",  # Slate gray
    "Background": "#F7F9FB",
    "Target_Map": {
        "homo_ev": "#2C5F8A",
        "lumo_ev": "#2A9D8F",
        "gap_ev": "#E9A03B"
    },
    "Metric_Map": {
        "MAE": "#4A5568",
        "RMSE": "#D3D3D3"
    }
}

FEATURE_DESCRIPTIONS = {
    "MolWt": "Molecular weight of the molecule.",
    "HeavyAtomCount": "Number of non-hydrogen atoms.",
    "NumHAcceptors": "Number of hydrogen bond acceptors.",
    "NumHDonors": "Number of hydrogen bond donors.",
    "TPSA": "Topological polar surface area.",
    "MolLogP": "Estimated hydrophobicity/lipophilicity.",
    "NumRotatableBonds": "Number of rotatable bonds related to molecular flexibility.",
    "FractionCSP3": "Fraction of sp3-hybridized carbon atoms.",
    "BertzCT": "Molecular complexity descriptor.",
    "LabuteASA": "Approximate molecular surface area.",
    "BalabanJ": "Topological connectivity index.",
    "RingCount": "Total number of rings.",
    "NumAromaticRings": "Number of aromatic rings.",
    "C_count": "Number of carbon atoms.",
    "H_count": "Number of hydrogen atoms.",
    "N_count": "Number of nitrogen atoms.",
    "O_count": "Number of oxygen atoms.",
    "F_count": "Number of fluorine atoms.",
    "S_count": "Number of sulfur atoms.",
    "Cl_count": "Number of chlorine atoms.",
    "Br_count": "Number of bromine atoms.",
    "I_count": "Number of iodine atoms.",
}

# 다국어 사전
TEXTS = {
    "KR": {
        "screening_title": "🔬 Molecular Screening App",
        "report_title": "📊 Machine Learning Model Report",
        "report_subtitle": "학습 데이터, 예측 결과 및 모델 성능 분석",
        "back_btn": "⬅️ 스크리닝 앱으로 돌아가기",
        "view_report_btn": "📊 View ML Model Report",
        "apply_filters": "Apply Screening Filters",
        "total_found": "Total Found",
        "overview_title": "📝 프로젝트 개요 (Project Overview)",
        "overview_text": "본 보고서는 QMUGS 기반 50,000개 고유 분자 데이터를 사용하여 HOMO, LUMO, HOMO-LUMO Gap 예측 모델을 학습하고, 여러 회귀 모델의 성능을 비교한 뒤, target별 최적 모델을 선정하는 과정을 요약합니다. 최종적으로 선정된 best model은 5,000개 molecular library의 orbital energy를 예측하는 데 사용되었습니다.",
        "training_data_title": "📁 1. 학습 데이터 요약 (Training Data Summary)",
        "features_title": "🧬 2. RDKit Descriptor 특성점 (Descriptor Features)",
        "workflow_title": "⚙️ 3. 모델 학습 및 평가 워크플로우 (Model Training Workflow)",
        "workflow_caption": "이 workflow는 단순히 하나의 모델을 사용한 것이 아니라, 여러 회귀 모델을 동일한 데이터 분할 조건에서 비교한 뒤 target별 best model을 선택하는 과정입니다.",
        "metrics_title": "📏 4. 모델 평가 지표 (Evaluation Metrics)",
        "metrics_mae_desc": "예측값이 실제값에서 평균적으로 얼마나 벗어나는지를 eV 단위로 나타냅니다. 낮을수록 좋습니다.",
        "metrics_rmse_desc": "큰 예측 오차에 더 민감한 지표입니다. 낮을수록 모델의 예측 오차가 작다는 의미입니다.",
        "metrics_r2_desc": "모델이 target property의 분산을 얼마나 잘 설명하는지를 나타냅니다. 1에 가까울수록 좋습니다.",
        "comparison_title": "📊 5. 모델 성능 비교 (Model Comparison)",
        "comparison_heatmap_r2_title": "모델별 R² 성능 Heatmap",
        "comparison_heatmap_error_title": "모델별 예측 오차 Heatmap",
        "comparison_heatmap_r2_desc": "이 heatmap은 각 모델이 HOMO, LUMO, HOMO-LUMO Gap을 얼마나 잘 예측했는지 R² 기준으로 비교한 것입니다. 색이 진할수록 해당 target에서 모델이 더 높은 설명력을 보였다는 의미입니다.",
        "comparison_heatmap_error_desc": "이 heatmap은 각 모델의 예측 오차를 RMSE 기준으로 비교한 것입니다. 값이 낮을수록 실제값에 더 가까운 예측을 수행했다는 의미입니다.",
        "comparison_heatmap_mae_title": "모델별 MAE Heatmap",
        "comparison_heatmap_mae_desc": "이 heatmap은 각 모델의 예측 오차를 MAE 기준으로 비교한 것입니다. 값이 낮을수록 평균 절대 오차가 작아 실제값에 더 가까운 예측을 수행했다는 의미입니다.",
        "comparison_bar_r2_title": "Target별 모델 R² 비교",
        "comparison_bar_r2_desc": "이 그래프는 각 target에서 어떤 모델이 가장 높은 R²를 보였는지 보여줍니다. HOMO, LUMO, Gap은 서로 다른 구조–물성 관계를 가질 수 있으므로 target별 best model이 다를 수 있습니다.",
        "comparison_bar_rmse_title": "Target별 모델 RMSE 비교",
        "comparison_bar_rmse_desc": "R²가 설명력을 보여준다면, RMSE는 실제 예측 오차의 크기를 eV 단위로 보여줍니다. 따라서 best model 선정 시 R²뿐 아니라 RMSE와 MAE도 함께 확인해야 합니다.",
        "ranking_table_title": "모델 성능 순위표 (Performance Ranking Table)",
        "ranking_table_desc": "이 표는 각 target별 모델 순위를 보여줍니다. 최종 best model은 target별로 가장 높은 R²를 중심으로 선정하되, MAE와 RMSE도 함께 고려했습니다.",
        "best_model_title": "🏆 6. 최적 모델 선정 (Best Model Selection)",
        "best_model_r2_title": "Target별 Best Model R²",
        "best_model_r2_desc": "이 그래프는 최종 예측 파이프라인에 사용된 target별 best model의 R² 성능을 요약합니다. LUMO 예측이 가장 높은 R²를 보였고, HOMO와 Gap도 screening-level 예측에 활용 가능한 수준의 설명력을 보였습니다.",
        "best_model_error_title": "Best Model의 MAE/RMSE 비교",
        "best_model_error_desc": "이 그래프는 최종 선택된 best model들의 예측 오차를 eV 단위로 비교합니다. MAE와 RMSE가 모두 낮을수록 실제 DFT target 값에 더 가까운 예측을 수행한 것입니다.",
        "best_margin_title": "Best vs Second-Best Comparison",
        "best_margin_desc": "이 표는 best model이 second-best model보다 얼마나 우수했는지를 보여줍니다. 성능 차이가 작을 경우 여러 모델이 유사한 예측력을 보였음을 의미합니다.",
        "quality_analysis_title": "📈 7. 예측 품질 분석 (Prediction Quality Analysis)",
        "parity_title": "Parity Plot (Actual vs Predicted)",
        "parity_desc_short": "Parity plot은 실제값과 예측값이 얼마나 잘 일치하는지 보여줍니다. 점들이 대각선에 가까울수록 모델이 실제 target 값을 잘 예측했다는 의미입니다.",
        "residual_title": "Residual Plot",
        "residual_desc_short": "Residual plot은 예측 오차가 특정 에너지 영역에서 체계적으로 커지는지 확인하는 그래프입니다. 점들이 0 근처에 고르게 분포할수록 편향이 작은 모델입니다.",
        "residual_dist_title": "Residual Distribution",
        "importance_title": "🔑 8. 특성 중요도 (Feature Importance Analysis)",
        "importance_desc_short": "Feature importance는 모델이 예측 과정에서 어떤 descriptor를 상대적으로 중요하게 사용했는지 보여줍니다. 이는 모델이 단순히 수치를 맞춘 것이 아니라, 분자량, 원소 조성 등 물리화학적 특성과 target property 사이의 상관관계를 학습했는지 해석하는 데 도움이 됩니다.",
        "importance_caution": "단, feature importance는 인과관계를 직접 의미하는 것은 아니며, 해당 모델이 예측에 활용한 상대적 중요도를 나타내는 지표입니다.",
        "library_application_title": "🌐 9. 5,000개 분자 라이브러리 적용 결과 (Application to Library)",
        "library_dist_desc": "이 그래프는 최종 best model을 5,000개 molecular library에 적용했을 때 얻어진 예측 결과입니다. 학습/검증이 끝난 모델을 실제 screening library에 적용한 결과입니다.",
        "library_scatter_desc": "이 산점도는 5,000개 후보 분자가 예측 HOMO/LUMO 공간에서 어떻게 분포하는지를 보여줍니다. 색상은 HOMO-LUMO Gap을 의미합니다.",
        "library_corr_desc": "이 그래프는 단순 분자 descriptor와 예측 orbital property 사이의 경향성을 보여줍니다.",
        "library_heatmap_desc": "Correlation heatmap은 descriptor와 예측 orbital property 사이의 선형 상관관계를 요약합니다.",
        "top_candidates_title": "🔝 주요 후보 물질 예시 (Example Top Candidates)",
        "interpretation_title": "💡 10. 최종 해석 및 제한 사항 (Interpretation & Limitation)",
        "interpretation_text": "전체 모델 비교 결과, LUMO 예측은 가장 높은 R²를 보였으며, HOMO와 Gap 예측도 대규모 후보군을 선별하기 위한 screening-level surrogate model로 활용 가능한 수준의 성능을 보였습니다.",
        "limitation_title": "⚠️ 제한 사항 (Limitation)",
        "limitation_text": "본 예측값은 정밀 DFT 계산을 대체하기 위한 값이 아니라, 대규모 molecular library에서 유망 후보를 빠르게 선별하기 위한 screening-level prediction입니다. 최종 후보 물질에 대해서는 추가적인 DFT 계산 또는 실험 검증이 필요합니다.",
    },
    "EN": {
        "screening_title": "🔬 Molecular Screening App",
        "report_title": "📊 Machine Learning Model Report",
        "report_subtitle": "Training Data, Prediction Results, and Performance Analysis",
        "back_btn": "⬅️ Back to Screening App",
        "view_report_btn": "📊 View ML Model Report",
        "apply_filters": "Apply Screening Filters",
        "total_found": "Total Found",
        "overview_title": "📝 Project Overview",
        "overview_text": "This report summarizes the development of machine-learning models for predicting HOMO, LUMO, and HOMO-LUMO gap using a QMUGS-derived dataset of 50,000 unique molecules. Multiple regression models were compared, the best model was selected independently for each target, and the final models were applied to predict orbital energies for a 5,000-molecule screening library.",
        "training_data_title": "📁 1. Training Data Summary",
        "features_title": "🧬 2. RDKit Descriptor Features",
        "workflow_title": "⚙️ 3. Model Training and Evaluation Workflow",
        "workflow_caption": "This workflow highlights that the final prediction pipeline was built by comparing multiple regression models under the same train/test split and selecting the best-performing model for each target.",
        "metrics_title": "📏 4. Evaluation Metrics",
        "metrics_mae_desc": "Represents the average deviation of predictions from actual values in eV. Lower is better.",
        "metrics_rmse_desc": "More sensitive to large prediction errors. Lower values indicate smaller overall errors.",
        "metrics_r2_desc": "Indicates how well the model explains the variance of the target property. Closer to 1 is better.",
        "comparison_title": "📊 5. Model Comparison",
        "comparison_heatmap_r2_title": "R² Performance Heatmap by Model and Target",
        "comparison_heatmap_error_title": "Prediction Error Heatmap by Model and Target",
        "comparison_heatmap_r2_desc": "This heatmap compares model performance using R² scores. Darker colors indicate higher explanatory power.",
        "comparison_heatmap_error_desc": "This heatmap compares prediction errors using RMSE. Lower values indicate closer proximity to actual values.",
        "comparison_heatmap_mae_title": "Model MAE Heatmap (Lower is Better)",
        "comparison_heatmap_mae_desc": "This heatmap compares model prediction error using MAE. Lower values indicate smaller average absolute error and therefore better agreement with the reference values.",
        "comparison_bar_r2_title": "Model R² Comparison by Target",
        "comparison_bar_r2_desc": "Shows which model achieved the highest R² for each target. Best models may vary by property.",
        "comparison_bar_rmse_title": "Model RMSE Comparison by Target",
        "comparison_bar_rmse_desc": "While R² shows explanatory power, RMSE shows the physical magnitude of errors in eV.",
        "ranking_table_title": "Performance Ranking Table",
        "ranking_table_desc": "Ranked list of models by R² score for each target property.",
        "best_model_title": "🏆 6. Best Model Selection",
        "best_model_r2_title": "Best Model R² by Target",
        "best_model_r2_desc": "Summary of R² performance for the final models used in the prediction pipeline.",
        "best_model_error_title": "MAE/RMSE of Best Models",
        "best_model_error_desc": "Comparison of prediction errors (eV) for the selected best models.",
        "best_margin_title": "Best vs Second-Best Comparison",
        "best_margin_desc": "Comparison showing the margin between the best and second-best models.",
        "quality_analysis_title": "📈 7. Prediction Quality Analysis",
        "parity_title": "Parity Plot (Actual vs Predicted)",
        "parity_desc_short": "Shows how closely predictions follow reference values. Alignment with the diagonal line indicates high accuracy.",
        "residual_title": "Residual Plot",
        "residual_desc_short": "Used to check for systematic bias. Uniform distribution around zero is ideal.",
        "residual_dist_title": "Residual Distribution",
        "importance_title": "🔑 8. Feature Importance Analysis",
        "importance_desc_short": "Shows which molecular descriptors contribute most to the model's predictions, helping interpret the structure-property relationship.",
        "importance_caution": "Note: Feature importance indicates relative model contribution, not direct causality.",
        "library_application_title": "🌐 9. Application to 5,000-Molecule Library",
        "library_dist_desc": "Predicted orbital energy distribution for the 5,000-molecule screening library using the final best models.",
        "library_scatter_desc": "Scatter plot showing the distribution of the screening set in HOMO/LUMO space.",
        "library_corr_desc": "Correlations between simple molecular descriptors and predicted orbital properties.",
        "library_heatmap_desc": "Summary of linear correlations between descriptors and predicted properties.",
        "top_candidates_title": "🔝 Example Top Candidates",
        "interpretation_title": "💡 10. Interpretation & Limitation",
        "interpretation_text": "LUMO prediction showed the highest performance, while HOMO and Gap models also achieved suitable accuracy for screening-level surrogate prediction.",
        "limitation_title": "⚠️ Limitation",
        "limitation_text": "These ML predictions are intended as screening-level surrogate models, not direct replacements for high-level DFT calculations. Final candidates require experimental or DFT validation.",
    }
}

# 페이지 설정
st.set_page_config(page_title="Molecular Screening App", layout="wide")

# 세션 상태 초기화
if "page" not in st.session_state: st.session_state["page"] = "screening"
if "lang" not in st.session_state: st.session_state["lang"] = "KR"
if "filters_applied" not in st.session_state: st.session_state["filters_applied"] = False
if "current_page" not in st.session_state: st.session_state["current_page"] = 1
if "items_per_page" not in st.session_state: st.session_state["items_per_page"] = 10

def toggle_lang():
    st.session_state.lang = "EN" if st.session_state.lang == "KR" else "KR"

# 작용기 패턴 및 도우미 함수들 (기존 유지)
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

def calculate_total_atoms(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol: return Chem.AddHs(mol).GetNumAtoms()
    except: pass
    return None

def calculate_rings(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol: return rdMolDescriptors.CalcNumRings(mol)
    except: pass
    return 0

def has_selected_functionalities(smiles, selected_patterns):
    if not selected_patterns: return True
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return False
        for pattern in selected_patterns:
            query = Chem.MolFromSmarts(pattern)
            if query and mol.HasSubstructMatch(query): return True
        return False
    except: return False

@st.cache_data
def load_csv(path):
    """CSV 파일을 로드하고 캐싱합니다."""
    if path is None or not os.path.exists(path): return None
    return pd.read_csv(path)

@st.cache_data
def load_json(path):
    """JSON 파일을 로드하고 캐싱합니다."""
    if path is None or not os.path.exists(path): return None
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data
def load_data(file_path):
    """메인 라이브러리 데이터를 로드하고 RDKit Descriptor를 계산하여 캐싱합니다."""
    start_time = time.time()
    if file_path is None or not os.path.exists(file_path): return None
    try:
        df = pd.read_csv(file_path)
        if "canonical_smiles" not in df.columns and "smiles" in df.columns:
            df["canonical_smiles"] = df["smiles"]
        
        # 부족한 기본 Descriptor 계산 (시각화용)
        if "total_atom_count" not in df.columns:
            df["total_atom_count"] = df["canonical_smiles"].apply(calculate_total_atoms)
        if "ring_count" not in df.columns:
            df["ring_count"] = df["canonical_smiles"].apply(calculate_rings)
        
        # 추가 Descriptor (LogP, TPSA 등) 가 없을 경우 계산
        def calc_more_desc(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol: return Descriptors.MolLogP(mol), Descriptors.TPSA(mol)
            except: pass
            return None, None

        if "mol_logp_rdkit" not in df.columns or "tpsa_rdkit" not in df.columns:
            logs_tpsas = df["canonical_smiles"].apply(calc_more_desc)
            if "mol_logp_rdkit" not in df.columns:
                df["mol_logp_rdkit"] = [x[0] for x in logs_tpsas]
            if "tpsa_rdkit" not in df.columns:
                df["tpsa_rdkit"] = [x[1] for x in logs_tpsas]

        elapsed = time.time() - start_time
        print(f"[LOG] load_data took {elapsed:.2f}s for {len(df)} molecules.")
        return df.dropna(subset=["total_atom_count", "ring_count"])
    except Exception as e:
        print(f"[ERR] Error in load_data: {e}")
        return None

def render_molecule_card(row):
    with st.container(border=True):
        c1, c2 = st.columns([1, 2.2])
        with c1:
            mol = Chem.MolFromSmiles(row["canonical_smiles"])
            if mol: st.image(Draw.MolToImage(mol, size=(180, 180)), use_container_width=True)
            else: st.write("No Image")
        with c2:
            display_name = row['iupac_name'] if pd.notna(row['iupac_name']) else 'Unnamed'
            st.markdown(f"<div style='white-space: normal; word-break: break-word; font-weight: bold; font-size: 1.0rem; margin-bottom: 4px;'>{display_name}</div>", unsafe_allow_html=True)
            st.markdown(f"<small>CID: {int(row['cid'])}</small>", unsafe_allow_html=True)
            smiles = row["canonical_smiles"]
            st.code(smiles[:45] + "..." if len(smiles) > 45 else smiles, language="text")
            info_html = (
                f"<div style='line-height:1.2;'><small>"
                f"⚖️ Molecular Weight: <b>{row['mol_wt_rdkit']:.1f}</b> | 💎 Heavy: <b>{int(row['heavy_atom_count'])}</b><br>"
                f"🏗️ Ring: <b>{int(row['ring_count'])}</b> (Aro: {int(row['aromatic_ring_count'])})<br>"
            )
            if "pred_homo_ev" in row:
                info_html += (
                    f"<span style='color: {SCIENTIFIC_COLORS['HOMO']};'>⚡ HOMO: <b>{row['pred_homo_ev']:.3f} eV</b></span> | "
                    f"<span style='color: {SCIENTIFIC_COLORS['LUMO']};'>⚡ LUMO: <b>{row['pred_lumo_ev']:.3f} eV</b></span><br>"
                    f"<span style='color: {SCIENTIFIC_COLORS['GAP']};'>↔️ Gap: <b>{row['pred_gap_ev']:.3f} eV</b></span>"
                )
            st.markdown(info_html + "</small></div>", unsafe_allow_html=True)

# --- 필터 동기화 함수들 (기존 유지) ---
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

def get_filter_values_synced(df, col_name, label, step=0.1):
    abs_min, abs_max = float(df[col_name].min()), float(df[col_name].max())
    if abs_min == abs_max: return None, None, None
    init_filter_state(df, col_name)
    st.sidebar.markdown(f"**{label}**")
    st.sidebar.slider(f"{label} Slider", abs_min, abs_max, key=f"{col_name}_slider", on_change=slider_callback, args=(col_name,), step=step, label_visibility="collapsed")
    c1, c2 = st.sidebar.columns(2)
    with c1: st.number_input("Min", key=f"{col_name}_min", on_change=input_callback, args=(col_name,), step=step)
    with c2: st.number_input("Max", key=f"{col_name}_max", on_change=input_callback, args=(col_name,), step=step)
    return st.session_state[f"{col_name}_min"], st.session_state[f"{col_name}_max"], (f"'{label}'의 Min이 Max보다 큽니다." if st.session_state[f"{col_name}_min"] > st.session_state[f"{col_name}_max"] else None)

# --- 렌더링 함수들 ---

def render_screening_page():
    L = TEXTS[st.session_state.lang]
    # 1. Top Row: Title and Report Button
    t_col1, t_col2 = st.columns([2, 1])
    with t_col1:
        st.title(L["screening_title"])
        subtitle = "ML-Powered Orbital Energy Prediction & Screening" if st.session_state.lang == "EN" else "머신러닝 기반 HOMO/LUMO 에너지 예측 및 스크리닝"
        st.markdown(f"<h4 style='color: #2C5F8A; font-weight: 400; margin-top: -15px; margin-bottom: 5px;'>{subtitle}</h4>", unsafe_allow_html=True)
    
    with t_col2:
        st.write(" ") 
        st.write(" ") 
        if st.button(L["view_report_btn"], use_container_width=True):
            st.session_state.page = "report"
            st.rerun()

    # 2. Hero Row: Large Visual (Hide if filters applied)
    if not st.session_state["filters_applied"]:
        v_col1, v_col2 = st.columns([1, 2])
        with v_col1:
            st.write("") # Placeholder for spacing where short_desc was
        
        with v_col2:
            # High-Fidelity Dynamic Atom Animation (Reflects 3D Render Style)
            molecule_html = """
            <style>
                .atom-scene {
                    width: 100%;
                    height: 450px;
                    background: #ffffff;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    perspective: 1500px;
                    margin-top: 50px;
                    margin-left: -160px; /* Shifted more left */
                    overflow: hidden;
                }
                .atom-wrapper {
                    position: relative;
                    width: 300px;
                    height: 300px;
                    transform-style: preserve-3d;
                }
                
                /* Realistic 3D Nucleus Cluster */
                .nucleus-cluster {
                    position: absolute;
                    width: 60px; height: 60px;
                    left: 120px; top: 120px;
                    transform-style: preserve-3d;
                    animation: nucleusFloat 6s infinite ease-in-out;
                }
                .nucleus-sphere {
                    position: absolute;
                    border-radius: 50%;
                    box-shadow: inset -5px -5px 15px rgba(0,0,0,0.2), 5px 5px 15px rgba(0,0,0,0.1);
                }
                .ns1 { width: 40px; height: 40px; background: radial-gradient(circle at 30% 30%, #ff9999, #e53e3e); left: 10px; top: 10px; z-index: 5; }
                .ns2 { width: 35px; height: 35px; background: radial-gradient(circle at 30% 30%, #99ccff, #3182ce); left: 25px; top: 5px; transform: translateZ(20px); }
                .ns3 { width: 35px; height: 35px; background: radial-gradient(circle at 30% 30%, #ff9999, #e53e3e); left: 5px; top: 25px; transform: translateZ(-20px); }
                .ns4 { width: 30px; height: 30px; background: radial-gradient(circle at 30% 30%, #99ccff, #3182ce); left: 20px; top: 20px; transform: translateZ(10px); }
    
                /* Sync-optimized Orbital System */
                .orbit-system {
                    position: absolute;
                    width: 300px; height: 300px;
                    transform-style: preserve-3d;
                }
                /* 3D Tilts */
                .tilt1 { transform: rotateX(75deg) rotateY(15deg); }
                .tilt2 { transform: rotateX(-45deg) rotateY(35deg); }
                .tilt3 { transform: rotateX(30deg) rotateY(80deg); }
    
                /* Rotating Ring with Fixed Electron (Perfect Sync) */
                .rotating-ring {
                    width: 100%; height: 100%;
                    border: 1.2px solid rgba(0, 191, 255, 0.3);
                    border-radius: 50%;
                    position: relative;
                    transform-style: preserve-3d;
                    animation: ringRotate 4s infinite linear;
                }
                .electron {
                    position: absolute;
                    width: 14px; height: 14px;
                    background: #ffffff;
                    border-radius: 50%;
                    box-shadow: 0 0 10px #fff, 0 0 20px #00bfff;
                    top: 50%; left: -7px; /* Perfectly centered on the line */
                    margin-top: -7px;
                }
                
                /* Variation in speed and direction */
                .r1 { animation-duration: 3s; }
                .r2 { animation-duration: 5s; animation-direction: reverse; }
                .r3 { animation-duration: 4s; }
    
                @keyframes ringRotate {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
                @keyframes nucleusFloat {
                    0%, 100% { transform: translateY(0px) rotateX(0deg); }
                    50% { transform: translateY(-8px) rotateX(5deg); }
                }
            </style>
            <div class="atom-scene">
                <div class="atom-wrapper">
                    <div class="nucleus-cluster">
                        <div class="nucleus-sphere ns1"></div>
                        <div class="nucleus-sphere ns2"></div>
                        <div class="nucleus-sphere ns3"></div>
                        <div class="nucleus-sphere ns4"></div>
                    </div>
                    <div class="orbit-system tilt1"><div class="rotating-ring r1"><div class="electron"></div></div></div>
                    <div class="orbit-system tilt2"><div class="rotating-ring r2"><div class="electron"></div></div></div>
                    <div class="orbit-system tilt3"><div class="rotating-ring r3"><div class="electron"></div></div></div>
                </div>
            </div>
            """
            components.html(molecule_html, height=450)

    df = load_data(DATA_PATH)
    if df is None:
        st.error("❌ Data missing.")
        return
        
    st.sidebar.info(f"Library: **{len(df)}** loaded")
    st.sidebar.header("🔍 Filters")
    all_filters, filter_errors = {}, []
    
    # 기본 필터들
    filter_sections = [
        ("📐 Size", [("mol_wt_rdkit", "Molecular Weight"), ("heavy_atom_count", "Heavy Atoms")]),
        ("⚛️ Atom Count", [
            ("total_atom_count", "Total Atom Count"),
            ("C_count", "C Count"),
            ("N_count", "N Count"),
            ("O_count", "O Count"),
            ("S_count", "S Count"),
            ("halogen_count", "Halogen Count")
        ]),
        ("🏗️ Structure", [("ring_count", "Ring Count"), ("aromatic_ring_count", "Aromatic Rings")]),
        ("⚡ Predicted Orbitals", [("pred_homo_ev", "HOMO (eV)"), ("pred_lumo_ev", "LUMO (eV)"), ("pred_gap_ev", "Gap (eV)")])
    ]
    for sec_name, configs in filter_sections:
        st.sidebar.subheader(sec_name)
        for col, lab in configs:
            if col in df.columns:
                # Orbital 에너지 예측값은 0.01 단위, 나머지는 정수(1.0) 단위로 설정
                step_val = 0.01 if "pred" in col else 1.0
                vmin, vmax, err = get_filter_values_synced(df, col, lab, step=step_val)
                if vmin is not None: all_filters[col] = (vmin, vmax)
                if err: filter_errors.append(err)

    if st.sidebar.button(L["apply_filters"], type="primary", use_container_width=True):
        st.session_state["filters_applied"] = True
        st.session_state["current_page"] = 1
        st.rerun()

    if st.session_state["filters_applied"]:
        filtered_df = df.copy()
        for col, (vmin, vmax) in all_filters.items():
            filtered_df = filtered_df[(filtered_df[col] >= vmin) & (filtered_df[col] <= vmax)]
        
        st.divider()
        st.subheader("🧪 Functionality Filters")
        selected_patterns = []
        fn_cols = st.columns(4)
        for i, (name, pattern) in enumerate(FUNCTIONAL_GROUPS.items()):
            with fn_cols[i % 4]:
                if st.checkbox(name, key=f"cb_{name}"): selected_patterns.append(pattern)
        
        if selected_patterns:
            filtered_df = filtered_df[filtered_df["canonical_smiles"].apply(lambda x: has_selected_functionalities(x, selected_patterns))]

        st.metric(L["total_found"], f"{len(filtered_df)} / {len(df)}")
        
        if not filtered_df.empty:
            items_per_page = st.session_state["items_per_page"]
            total_pages = (len(filtered_df) - 1) // items_per_page + 1
            start_idx = (st.session_state["current_page"] - 1) * items_per_page
            page_data = filtered_df.iloc[start_idx : start_idx + items_per_page]
            
            for i in range(0, len(page_data), 2):
                row_cols = st.columns(2)
                with row_cols[0]: render_molecule_card(page_data.iloc[i])
                if i + 1 < len(page_data):
                    with row_cols[1]: render_molecule_card(page_data.iloc[i+1])
            
            # 페이지네이션
            _, c_pag, _ = st.columns([1, 4, 1])
            with c_pag:
                nav = st.columns([1, 1, 2, 1])
                with nav[1]:
                    if st.button("⬅️", disabled=st.session_state.current_page <= 1):
                        st.session_state.current_page -= 1
                        st.rerun()
                with nav[2]: st.markdown(f"<center>Page {st.session_state.current_page} / {total_pages}</center>", unsafe_allow_html=True)
                with nav[3]:
                    if st.button("➡️", disabled=st.session_state.current_page >= total_pages):
                        st.session_state.current_page += 1
                        st.rerun()

# --- 리포트 섹션 헬퍼 및 렌더링 함수들 ---

def render_step_card(step_num, title, description):
    """워크플로우 단계별 정보를 카드 레이아웃으로 표시합니다."""
    st.markdown(f"""
    <div style="padding: 1.2rem; border-radius: 0.8rem; background-color: #f8f9fa; border-left: 5px solid #2C5F8A; margin-bottom: 1rem; height: 100%;">
        <h4 style="margin: 0; color: #2C5F8A; font-size: 1.1rem;">Step {step_num}</h4>
        <h5 style="margin: 0.3rem 0; color: #1A202C;">{title}</h5>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #4A5568; line-height: 1.4;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

def render_training_data_summary(L):
    st.subheader(L["training_data_title"])
    c1, c2, c3 = st.columns(3)
    with c1: st.info("**Source**: QMUGS (50k Unique Mols)")
    with c2: st.info("**Features**: 22 RDKit Descriptors")
    with c3: st.info("**Targets**: HOMO, LUMO, Gap")

def render_descriptor_table(L):
    st.subheader(L["features_title"])
    with st.expander("View Descriptor Details"):
        data = [{"Feature": f, "Description": FEATURE_DESCRIPTIONS.get(f, "Molecular descriptor.")} for f in FEATURE_DESCRIPTIONS.keys()]
        st.table(pd.DataFrame(data))

def render_model_workflow(L):
    st.subheader(L["workflow_title"])
    st.markdown(L["workflow_caption"])
    
    w_cols = st.columns(5)
    steps = [
        ("Data Preparation", "50,000 QMUGS 데이터 로드 및 Descriptor 계산"),
        ("Model Comparison", "XGBoost, LightGBM, Random Forest 등 6개 모델 평가"),
        ("Hyperparameter Tuning", "각 모델별 최적의 파라미터 탐색 (CV 수행)"),
        ("Best Model Selection", "Target별 가장 낮은 RMSE/높은 R² 모델 선정"),
        ("Library Prediction", "선정된 모델을 5,000개 라이브러리에 적용")
    ]
    for i, (title, desc) in enumerate(steps):
        with w_cols[i]: render_step_card(i+1, title, desc)

def render_evaluation_metrics(L):
    st.subheader(L["metrics_title"])
    m1, m2, m3 = st.columns(3)
    with m1:
        with st.container(border=True):
            st.markdown(f"**MAE** (Mean Absolute Error)")
            st.write(L["metrics_mae_desc"])
    with m2:
        with st.container(border=True):
            st.markdown(f"**RMSE** (Root Mean Squared Error)")
            st.write(L["metrics_rmse_desc"])
    with m3:
        with st.container(border=True):
            st.markdown(f"**R²** (Coefficient of Determination)")
            st.write(L["metrics_r2_desc"])

def render_model_comparison(L):
    st.subheader(L["comparison_title"])
    path = BASE_DIR / "models" / "qmugs_model_comparison.csv"
    df = load_csv(str(path))
    if df is None:
        st.warning("Comparison data missing.")
        return
    
    # 1. Heatmap Comparison Helpers
    def show_compact_matplotlib_fig(fig, width=460):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        c1, c2, c3 = st.columns([1, 1.2, 1])
        with c2:
            st.image(buf, width=width)
        plt.close(fig)

    def render_compact_metric_heatmap(pivot_df, title, cmap, value_fmt=".3f"):
        with plt.rc_context({
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
        }):
            fig, ax = plt.subplots(figsize=(3.8, 2.8), dpi=120)
            im = ax.imshow(pivot_df.values, cmap=cmap, aspect="auto")
            ax.set_title(title, fontsize=8, fontweight="normal", pad=6)
            ax.set_xlabel("target", fontsize=7, fontweight="normal")
            ax.set_ylabel("model", fontsize=7, fontweight="normal")
            ax.set_xticks(np.arange(len(pivot_df.columns)))
            ax.set_xticklabels(pivot_df.columns, rotation=35, ha="right", fontsize=6, fontweight="normal")
            ax.set_yticks(np.arange(len(pivot_df.index)))
            ax.set_yticklabels(pivot_df.index, fontsize=6, fontweight="normal")
            ax.tick_params(axis="both", labelsize=6)

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("normal")

            values = pivot_df.values
            mean_value = np.nanmean(values)

            for i in range(pivot_df.shape[0]):
                for j in range(pivot_df.shape[1]):
                    value = values[i, j]
                    ax.text(j, i, format(value, value_fmt), ha="center", va="center", fontsize=6, fontweight="normal", alpha=0.85, color="white" if value > mean_value else "black")

            cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            cbar.ax.tick_params(labelsize=6)
            for label in cbar.ax.get_yticklabels():
                label.set_fontweight("normal")
            fig.tight_layout()
            return fig

    # 1-1. R2 Heatmap
    st.markdown(f"#### {L['comparison_heatmap_r2_title']}")
    st.write(L["comparison_heatmap_r2_desc"])
    pivot_r2 = df.pivot(index="model", columns="target", values="R2")
    fig_r2 = render_compact_metric_heatmap(pivot_r2, "Model R² Performance", "viridis")
    show_compact_matplotlib_fig(fig_r2, width=460)
    
    # 1-2. RMSE Heatmap
    st.markdown(f"#### {L['comparison_heatmap_error_title']}")
    st.write(L["comparison_heatmap_error_desc"])
    pivot_rmse = df.pivot(index="model", columns="target", values="RMSE")
    fig_rmse = render_compact_metric_heatmap(pivot_rmse, "Model RMSE Performance", "Reds")
    show_compact_matplotlib_fig(fig_rmse, width=460)

    # 1-3. MAE Heatmap
    st.markdown(f"#### {L['comparison_heatmap_mae_title']}")
    st.write(L["comparison_heatmap_mae_desc"])
    pivot_mae = df.pivot(index="model", columns="target", values="MAE")
    fig_mae = render_compact_metric_heatmap(pivot_mae, "Model MAE Performance", "Oranges")
    show_compact_matplotlib_fig(fig_mae, width=460)
    
    # 2. Bar Chart Comparison
    st.markdown(f"#### {L['comparison_bar_r2_title']}")
    st.write(L["comparison_bar_r2_desc"])
    fig_bar_r2 = px.bar(df, x="model", y="R2", color="target", barmode="group", color_discrete_map=SCIENTIFIC_COLORS["Target_Map"], template="plotly_white")
    st.plotly_chart(fig_bar_r2, use_container_width=True)
    
    # 3. Ranking Table
    st.markdown(f"#### {L['ranking_table_title']}")
    st.write(L["ranking_table_desc"])
    st.dataframe(df.sort_values(["target", "R2"], ascending=[True, False]), hide_index=True, use_container_width=True)

def render_best_model_selection(L):
    st.subheader(L["best_model_title"])
    path_summary = BASE_DIR / "models" / "qmugs_best_model_summary.json"
    best_json = load_json(str(path_summary))
    if not best_json:
        st.warning("Best model summary missing.")
        return
    best_data = best_json.get("best_models", {})
    
    # 1. Best Model Cards
    b_cols = st.columns(3)
    targets = [("homo_ev", "HOMO"), ("lumo_ev", "LUMO"), ("gap_ev", "Gap")]
    plot_rows = []
    for i, (k, lab) in enumerate(targets):
        if k in best_data:
            info = best_data[k]
            plot_rows.append({"Target": lab, "R2": info["R2"], "MAE": info["MAE"], "RMSE": info["RMSE"], "Model": info.get("model_name", "Unknown")})
            with b_cols[i]:
                with st.container(border=True):
                    st.markdown(f"<h4 style='color:{SCIENTIFIC_COLORS['Target_Map'][k]}; margin-top:0;'>{lab}</h4>", unsafe_allow_html=True)
                    st.write(f"**Model**: {info.get('model_name', 'Model')}")
                    st.metric("R² Score", f"{info['R2']:.4f}")
                    st.write(f"MAE: {info['MAE']:.4f} eV")
                    st.write(f"RMSE: {info['RMSE']:.4f} eV")
    
    df_best = pd.DataFrame(plot_rows)
    
    # 2. Comparison Plots
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### {L['best_model_r2_title']}")
        st.write(L["best_model_r2_desc"])
        fig_best_r2 = px.bar(df_best, x="Target", y="R2", color="Target", color_discrete_map={"HOMO": SCIENTIFIC_COLORS["HOMO"], "LUMO": SCIENTIFIC_COLORS["LUMO"], "Gap": SCIENTIFIC_COLORS["GAP"]}, template="plotly_white")
        st.plotly_chart(fig_best_r2, use_container_width=True)
    with c2:
        st.markdown(f"#### {L['best_model_error_title']}")
        st.write(L["best_model_error_desc"])
        df_m = df_best.melt(id_vars="Target", value_vars=["MAE", "RMSE"])
        fig_best_err = px.bar(df_m, x="Target", y="value", color="variable", barmode="group", color_discrete_sequence=[SCIENTIFIC_COLORS["Gray"], "#D3D3D3"], template="plotly_white")
        st.plotly_chart(fig_best_err, use_container_width=True)

    # 3. Margin Analysis (Optional Table)
    st.markdown(f"#### {L['best_margin_title']}")
    st.write(L["best_margin_desc"])
    st.table(df_best[["Target", "Model", "R2", "RMSE"]])

def render_parity_residual_analysis(L):
    st.subheader(L["quality_analysis_title"])
    targets = [("homo_ev", "HOMO"), ("lumo_ev", "LUMO"), ("gap_ev", "Gap")]
    tabs = st.tabs([t[1] for t in targets])
    
    for i, (k, lab) in enumerate(targets):
        with tabs[i]:
            path = str(BASE_DIR / "models" / f"qmugs_test_predictions_{k}.csv")
            df_full = load_csv(path)
            if df_full is not None:
                df = df_full.sample(min(2000, len(df_full)))
                st.markdown(f"#### {lab} - {L['parity_title']}")
                st.write(L["parity_desc_short"])
                
                c1, c2 = st.columns(2)
                with c1:
                    fig_p = px.scatter(df, x="actual", y="predicted", opacity=0.4, color_discrete_sequence=[SCIENTIFIC_COLORS["Target_Map"][k]], title=f"Parity: {lab}", template="plotly_white")
                    fig_p.add_shape(type="line", x0=df["actual"].min(), y0=df["actual"].min(), x1=df["actual"].max(), y1=df["actual"].max(), line=dict(color="Red", dash="dash"))
                    st.plotly_chart(fig_p, use_container_width=True)
                with c2:
                    fig_r = px.scatter(df, x="predicted", y="residual", opacity=0.4, color_discrete_sequence=[SCIENTIFIC_COLORS["Gray"]], title=f"Residual: {lab}", template="plotly_white")
                    fig_r.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_r, use_container_width=True)
                
                st.markdown(f"#### {lab} - {L['residual_dist_title']}")
                fig_dist = px.histogram(df, x="residual", nbins=50, color_discrete_sequence=[SCIENTIFIC_COLORS["Gray"]], title=f"Residual Dist: {lab}", template="plotly_white")
                st.plotly_chart(fig_dist, use_container_width=True)
            else: st.warning(f"Validation data for {lab} missing.")

def render_feature_importance(L):
    st.subheader(L["importance_title"])
    st.write(L["importance_desc_short"])
    
    targets = [("homo_ev", "HOMO"), ("lumo_ev", "LUMO"), ("gap_ev", "Gap")]
    cols = st.columns(3)
    for i, (k, lab) in enumerate(targets):
        with cols[i]:
            path = str(BASE_DIR / "models" / f"qmugs_feature_importance_{k}.csv")
            df_imp = load_csv(path)
            if df_imp is not None:
                df_imp = df_imp.head(12)
                fig = px.bar(df_imp, x="importance", y="feature", orientation="h", title=f"Importance: {lab}", color_discrete_sequence=[SCIENTIFIC_COLORS["Target_Map"][k]], template="plotly_white")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning(f"Importance for {lab} missing.")
    st.caption(L["importance_caution"])

def render_library_application(L):
    st.subheader(L["library_application_title"])
    df = load_data(DATA_PATH)
    if df is None or "pred_homo_ev" not in df.columns:
        st.warning("Library prediction results not found.")
        return

    # 1. Distribution
    st.write(L["library_dist_desc"])
    d_cols = st.columns(3)
    for i, (col, lab) in enumerate([("pred_homo_ev", "HOMO"), ("pred_lumo_ev", "LUMO"), ("pred_gap_ev", "Gap")]):
        with d_cols[i]:
            fig = px.histogram(df, x=col, title=f"Library {lab} Dist (eV)", color_discrete_sequence=[SCIENTIFIC_COLORS[lab.upper()]], template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # 2. Correlation Scatter
    st.write(L["library_scatter_desc"])
    fig_scat = px.scatter(df, x="pred_homo_ev", y="pred_lumo_ev", color="pred_gap_ev", color_continuous_scale="Viridis", title="Library HOMO vs LUMO Map", hover_data=["cid", "iupac_name"], template="plotly_white")
    st.plotly_chart(fig_scat, use_container_width=True)

    # 3. Structure-Property Heatmap
    st.write(L["library_heatmap_desc"])
    potential_cols = ["mol_wt_rdkit", "mol_logp_rdkit", "tpsa_rdkit", "heavy_atom_count", "ring_count", "aromatic_ring_count", "pred_homo_ev", "pred_lumo_ev", "pred_gap_ev"]
    corr_cols = [c for c in potential_cols if c in df.columns]
    if len(corr_cols) > 1:
        corr_df = df[corr_cols].corr()
        fig_heat = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", title="Feature-Property Correlation", template="plotly_white")
        fig_heat.update_layout(coloraxis_colorbar_x=1.0, coloraxis_colorbar_thickness=20, width=800)
        st.plotly_chart(fig_heat, use_container_width=False)

    # 4. Top Candidates
    st.subheader(L["top_candidates_title"])
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        st.write("**Smallest Predicted Gap (eV)**")
        st.dataframe(df.nsmallest(10, "pred_gap_ev")[["cid", "pred_homo_ev", "pred_lumo_ev", "pred_gap_ev"]], hide_index=True)
    with t_c2:
        st.write("**Lowest Predicted LUMO (eV)**")
        st.dataframe(df.nsmallest(10, "pred_lumo_ev")[["cid", "pred_homo_ev", "pred_lumo_ev", "pred_gap_ev"]], hide_index=True)

def render_final_interpretation(L):
    st.subheader(L["interpretation_title"])
    st.info(L["interpretation_text"])
    st.divider()
    st.subheader(L["limitation_title"])
    st.warning(L["limitation_text"])

def render_model_report_page():
    L = TEXTS[st.session_state.lang]
    h1, h2 = st.columns([3, 1.2])
    with h1: st.title(L["report_title"])
    with h2:
        st.write("")
        c_b1, c_b2 = st.columns(2)
        with c_b1:
            if st.button("🌐 EN/KR", on_click=toggle_lang, use_container_width=True): pass
        with c_b2:
            if st.button(L["back_btn"], use_container_width=True):
                st.session_state.page = "screening"
                st.rerun()

    st.markdown(f"### {L['report_subtitle']}")
    st.divider()

    # 1. Overview
    st.subheader(L["overview_title"])
    st.write(L["overview_text"])
    
    # 2. Data & Features
    render_training_data_summary(L)
    render_descriptor_table(L)
    st.divider()

    # 3. Workflow
    render_model_workflow(L)
    st.divider()

    # 4. Metrics
    render_evaluation_metrics(L)
    st.divider()

    # 5. Model Comparison
    render_model_comparison(L)
    st.divider()

    # 6. Best Model Selection
    render_best_model_selection(L)
    st.divider()

    # 7. Parity & Residual
    render_parity_residual_analysis(L)
    st.divider()

    # 8. Importance
    render_feature_importance(L)
    st.divider()

    # 9. Application
    render_library_application(L)
    st.divider()

    # 10. Interpretation
    render_final_interpretation(L)

def main():
    if st.session_state.page == "screening": render_screening_page()
    else: render_model_report_page()

if __name__ == "__main__": main()
