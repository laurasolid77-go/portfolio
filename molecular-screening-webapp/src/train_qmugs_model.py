import os
import pandas as pd
import numpy as np
import pickle
import json
import time
from pathlib import Path
from datetime import datetime

# Scikit-learn imports
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor, 
    ExtraTreesRegressor, 
    GradientBoostingRegressor, 
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Safe imports for optional boosting libraries
XGB_AVAILABLE = False
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    pass

LGBM_AVAILABLE = False
try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    pass

# 프로젝트 루트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent

def load_data(file_path):
    """데이터를 로드하고 결측치를 처리합니다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {file_path}")
    
    print(f"\n[1] 데이터 로드 중: {file_path}")
    df = pd.read_csv(file_path)
    print(f"    - 전체 데이터 수: {len(df)}")
    
    return df

def get_feature_columns(df):
    """Feature 컬럼과 Target 컬럼을 분리합니다."""
    targets = ['homo_ev', 'lumo_ev', 'gap_ev']
    
    # 제외할 컬럼들
    exclude_cols = [
        'smiles', 'canonical_smiles', 
        'homo_ev', 'lumo_ev', 'gap_ev', 
        'homo_hartree', 'lumo_hartree', 'gap_hartree'
    ]
    
    features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"    - Feature 수: {len(features)}")
    print(f"    - Target 리스트: {targets}")
    
    return features, targets

def get_available_models():
    """사용 가능한 모델 목록과 파라미터를 반환합니다."""
    models = {
        'RandomForestRegressor': RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        'ExtraTreesRegressor': ExtraTreesRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        'GradientBoostingRegressor': GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=42
        ),
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ]),
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])
    }
    
    print(f"\n[2] 모델 구성 확인")
    
    if XGB_AVAILABLE:
        models['XGBRegressor'] = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            objective='reg:squarederror', random_state=42, 
            n_jobs=-1, tree_method='hist'
        )
        print("    - XGBoost: 사용 가능")
    else:
        print("    - XGBoost: 설치되지 않음 (pip install xgboost 로 설치 가능)")
        
    if LGBM_AVAILABLE:
        models['LGBMRegressor'] = LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=-1,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
        print("    - LightGBM: 사용 가능")
    else:
        print("    - LightGBM: 설치되지 않음 (pip install lightgbm 으로 설치 가능)")
        
    print(f"    - 사용 가능한 모델 수: {len(models)}")
    return models

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """모델을 학습하고 성능을 평가합니다."""
    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return mae, rmse, r2, elapsed_time, y_pred

def train_and_compare_models(df, features, targets):
    """모든 Target과 모델에 대해 비교 실험을 진행합니다."""
    models_dict = get_available_models()
    
    comparison_results = []
    best_models = {}
    rf_models = {}
    
    # 데이터 전처리: 결측치 제거
    if df[features + targets].isnull().any().any():
        initial_count = len(df)
        df = df.dropna(subset=features + targets)
        print(f"\n[!] 결측치가 발견되어 제거되었습니다: {initial_count} -> {len(df)}")

    X = df[features]
    
    for target in targets:
        print(f"\n>>> Target: {target} 학습 시작")
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        target_results = []
        
        for name, model_template in models_dict.items():
            print(f"    - 학습 중: {name}...", end=" ", flush=True)
            try:
                model = clone(model_template)
                
                mae, rmse, r2, sec, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
                print(f"완료 ({sec:.1f}초) | R2: {r2:.4f}")
                
                res = {
                    'target': target,
                    'model': name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'time_sec': sec,
                    'trained_model': model,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                target_results.append(res)
                comparison_results.append(res)
                
            except Exception as e:
                print(f"실패! ({e})")
        
        # Best model 선정 (R2 기준)
        if target_results:
            target_df = pd.DataFrame(target_results)
            best_idx = target_df['R2'].idxmax()
            best_info = target_results[best_idx]
            
            print(f"    => {target} Best Model: {best_info['model']} (R2: {best_info['R2']:.4f})")
            
            best_models[target] = {
                'model_name': best_info['model'],
                'model_obj': best_info['trained_model'],
                'metrics': best_info,
                'y_test': best_info['y_test'],
                'y_pred': best_info['y_pred']
            }
            
            for res in target_results:
                if res['model'] == 'RandomForestRegressor':
                    rf_models[target] = res['trained_model']

    return pd.DataFrame(comparison_results), best_models, rf_models, features

def save_outputs(comparison_df, best_models, rf_models, features, total_samples):
    """결과물(모델, CSV, JSON, Parity, Importance)을 저장합니다."""
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature 리스트 저장
    feature_path = models_dir / "qmugs_feature_columns.pkl"
    with open(feature_path, 'wb') as f:
        pickle.dump(features, f)
    
    # 2. 비교 결과 CSV 저장
    csv_path = models_dir / "qmugs_model_comparison.csv"
    # trained_model 등 객체 컬럼 제외하고 저장
    save_cols = ['target', 'model', 'MAE', 'RMSE', 'R2', 'time_sec']
    comparison_df[save_cols].to_csv(csv_path, index=False)
    
    # 3. Best 모델 요약 정보
    summary_data = {
        'total_samples': total_samples,
        'feature_count': len(features),
        'train_test_split': '8:2',
        'random_state': 42,
        'best_models': {}
    }
    
    for target, info in best_models.items():
        # Best 모델 저장
        best_path = models_dir / f"qmugs_{target}_best.pkl"
        with open(best_path, 'wb') as f:
            pickle.dump(info['model_obj'], f)
        
        # Parity 데이터 저장 (Actual vs Predicted)
        parity_df = pd.DataFrame({
            'actual': info['y_test'],
            'predicted': info['y_pred'],
            'residual': info['y_test'] - info['y_pred']
        })
        parity_path = models_dir / f"qmugs_test_predictions_{target}.csv"
        parity_df.to_csv(parity_path, index=False)
        
        # Feature Importance 저장
        try:
            model = info['model_obj']
            importance = None
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif isinstance(model, Pipeline):
                # Ridge/LR Pipeline
                final_step = model.steps[-1][1]
                if hasattr(final_step, 'coef_'):
                    importance = np.abs(final_step.coef_)
            
            if importance is not None:
                imp_df = pd.DataFrame({
                    'feature': features,
                    'importance': importance
                }).sort_values(by='importance', ascending=False)
                imp_path = models_dir / f"qmugs_feature_importance_{target}.csv"
                imp_df.to_csv(imp_path, index=False)
        except Exception as e:
            print(f"    [!] {target} Feature Importance 저장 실패: {e}")
            
        # RF 모델 저장 (호환성용)
        if target in rf_models:
            rf_path = models_dir / f"qmugs_{target}_rf.pkl"
            with open(rf_path, 'wb') as f:
                pickle.dump(rf_models[target], f)
            
        # Summary 데이터 구성
        summary_data['best_models'][target] = {
            'model_name': info['model_name'],
            'MAE': info['metrics']['MAE'],
            'RMSE': info['metrics']['RMSE'],
            'R2': info['metrics']['R2']
        }
        
    # 4. JSON 요약 저장
    json_path = models_dir / "qmugs_best_model_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n[3] 저장 완료")
    print(f"    - 저장된 파일 목록:")
    print(f"      * {csv_path.name}")
    print(f"      * {json_path.name}")
    for target in best_models.keys():
        print(f"      * qmugs_test_predictions_{target}.csv")
        print(f"      * qmugs_feature_importance_{target}.csv")

def main():
    print("="*60)
    print("QMugs Molecular Orbitals Model Training & Comparison (Advanced)")
    print("="*60)
    
    input_file = BASE_DIR / "data" / "processed" / "qmugs_features.csv"
    
    try:
        df = load_data(input_file)
        features, targets = get_feature_columns(df)
        comparison_df, best_models, rf_models, features = train_and_compare_models(df, features, targets)
        save_outputs(comparison_df, best_models, rf_models, features, len(df))
        print(f"\n[성공] 모든 프로세스가 완료되었습니다.")
        
    except Exception as e:
        print(f"\n[에러] 프로세스 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
