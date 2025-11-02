"""
Self-Healing RAN Orchestrator
Coordinates all 8 agents for end-to-end self-healing workflow
"""
import os
import json
import time
import warnings
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.inspection import permutation_importance

# Import all agents
from agents.ingestion_agent import IngestionQualityAgent
from agents.detector_agent import DetectorAgent
from agents.pattern_profiler_agent import PatternProfilerAgent
from agents.rca_agent import RCAAagent
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.verifier_agent import VerifierAgent
from agents.knowledge_curator import KnowledgeCurator

warnings.filterwarnings('ignore')

# Try SHAP (optional)
try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False
    shap = None

# Configuration
DATA_PATH = Path('cell_kpi_anomalies.csv')
MODEL_ID = "us.amazon.nova-lite-v1:0"  # Amazon Nova Lite
AWS_REGION = "us-east-1"

# LLM usage configuration
USE_LLM_FOR = {
    'ingestion': True,
    'detection': True,
    'profiling': True,
    'rca': True,
    'planning': True,
    'execution': True,
    'verification': True,
    'curation': True,
}

def load_and_prepare_data():
    """Load and engineer features from data"""
    print("📊 Loading and preparing data...")
    
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['cell_name','timestamp']).reset_index(drop=True)
    
    # Derived KPIs
    if {'RRC.ConnEstabSucc.Sum','RRC.ConnEstabAtt.Sum'}.issubset(df.columns):
        df['RRC_SuccRate'] = (df['RRC.ConnEstabSucc.Sum'] / df['RRC.ConnEstabAtt.Sum']).replace([np.inf,-np.inf], np.nan)
    else:
        df['RRC_SuccRate'] = np.nan
    
    # Feature columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_cols = ['is_anomaly']
    feat_cols = [c for c in num_cols if c not in label_cols]
    
    # Per-cell baselines
    baselines = (df[~df['is_anomaly']]
                 .groupby('cell_name')[feat_cols]
                 .median(numeric_only=True))
    
    # Attach baseline deltas
    key_kpis = ['DRB.UEThpDl','DRB.UEThpUl','RRU.PrbTotDl','RRU.PrbTotUl','DRB.UECqiDl','DRB.UEBlerDl','RRC_SuccRate','PEE.AvgPower','Viavi.PEE.EnergyEfficiency']
    for k in key_kpis:
        if k in feat_cols:
            df[f'{k}__delta_pct_vs_cell_median'] = df.apply(
                lambda r: ((r.get(k,np.nan) - baselines.loc[r['cell_name'],k]) / (baselines.loc[r['cell_name'],k] + 1e-9) * 100) if r['cell_name'] in baselines.index else np.nan,
                axis=1
            )
            feat_cols.append(f'{k}__delta_pct_vs_cell_median')
    
    print(f"✅ Data loaded: {len(df):,} rows, {df['cell_name'].nunique()} cells, {len(feat_cols)} features")
    
    return df, feat_cols, baselines

def train_detector(df, feat_cols):
    """Train anomaly detection model"""
    print("\n🤖 Training anomaly detector...")
    
    X = df[feat_cols].fillna(0.0)
    y = df['is_anomaly'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:,1]
    print(classification_report(y_test, pred, digits=3))
    
    try:
        auc = roc_auc_score(y_test, proba)
        print(f"ROC-AUC: {auc:.3f}")
    except Exception:
        pass
    
    # SHAP explainer
    explainer = None
    if HAVE_SHAP:
        try:
            explainer = shap.TreeExplainer(clf)
            print("✅ SHAP explainer initialized")
        except Exception as e:
            print(f"⚠️  SHAP initialization failed: {e}")
    
    return clf, explainer

def initialize_agents(feat_cols, baselines, clf, explainer):
    """Initialize all 8 agents"""
    print("\n🤖 Initializing agents...")
    
    agents = {
        'ingestor': IngestionQualityAgent(
            feat_cols=feat_cols,
            region_name=AWS_REGION,
            model_id=MODEL_ID
        ),
        'detector': DetectorAgent(
            model=clf,
            feat_cols=feat_cols,
            region_name=AWS_REGION,
            model_id=MODEL_ID
        ),
        'profiler': PatternProfilerAgent(
            baselines=baselines,
            feat_cols=feat_cols,
            shap_explainer=explainer if HAVE_SHAP else None,
            region_name=AWS_REGION,
            model_id=MODEL_ID
        ),
        'rca_agent': RCAAagent(
            region_name=AWS_REGION,
            model_id=MODEL_ID
        ),
        'planner': PlannerAgent(
            region_name=AWS_REGION,
            model_id=MODEL_ID
        ),
        'executor': ExecutorAgent(
            region_name=AWS_REGION,
            model_id=MODEL_ID
        ),
        'verifier': VerifierAgent(
            baselines=baselines,
            region_name=AWS_REGION,
            model_id=MODEL_ID
        ),
        'curator': KnowledgeCurator(
            region_name=AWS_REGION,
            model_id=MODEL_ID
        )
    }
    
    llm_count = sum(USE_LLM_FOR.values())
    print(f"✅ All agents initialized")
    print(f"   🤖 Model: {MODEL_ID}")
    print(f"   🌍 Region: {AWS_REGION}")
    print(f"   📊 LLM Usage: {llm_count}/8 agents enabled")
    
    return agents

def process_cases(df, agents, num_cases=5):
    """Process anomaly cases through the self-healing workflow"""
    print(f"\n🔄 Processing {num_cases} anomaly cases...")
    
    # Select candidate anomalies
    candidates = df[df['is_anomaly']].copy()
    candidates['slot'] = candidates['timestamp'].dt.floor('15min')
    candidates = candidates.drop_duplicates(['cell_name','slot']).head(num_cases)
    
    results = []
    
    for idx, (_, row) in enumerate(candidates.iterrows(), 1):
        cell = row['cell_name']
        t0 = row['timestamp']
        
        print(f"\n{'='*70}")
        print(f"Case {idx}/{len(candidates)}: {cell} @ {t0}")
        print(f"{'='*70}")
        
        # Build batch
        batch = df[(df['cell_name']==cell) & (df['timestamp'].between(t0 - timedelta(minutes=5), t0))]
        if batch.empty:
            continue
        
        # 1. Ingestion
        print("1️⃣  Ingestion Quality Check...")
        cleaned_result = agents['ingestor'].process(batch)
        cleaned_batch = cleaned_result['clean']
        print(f"   Quality: {cleaned_result['quality']['severity']} ({cleaned_result['quality']['missing_pct']:.1f}% missing)")
        
        # 2. Detection
        print("2️⃣  Anomaly Detection...")
        det = agents['detector'].detect(cleaned_batch)
        if not det['is_anomaly_any']:
            print("   No anomalies detected, skipping")
            continue
        print(f"   Detected: {det['anomaly_count']} anomalies (max prob: {det['max_probability']:.2f})")
        
        # 3. Profile
        print("3️⃣  Pattern Profiling...")
        r = batch.iloc[-1]
        fp = agents['profiler'].fingerprint(r)
        print(f"   Fingerprint: {len(fp)} signals")
        
        # 4. RCA
        print("4️⃣  Root Cause Analysis...")
        rca = agents['rca_agent'].diagnose(fp)
        print(f"   Cause: {rca['primary_cause']} (confidence: {rca['confidence']:.2f})")
        
        # 5. Plan
        print("5️⃣  Remediation Planning...")
        plan = agents['planner'].plan(rca, cell)
        print(f"   Actions: {len(plan['actions'])} planned")
        
        # 6. Execute
        print("6️⃣  Execution...")
        exec_res = agents['executor'].apply(plan)
        print(f"   Status: {exec_res['status']}")
        
        # 7. Verify
        print("7️⃣  Verification...")
        after = df[(df['cell_name']==cell) & (df['timestamp'].between(t0, t0 + timedelta(minutes=15)))]
        after_row = after.iloc[-1] if not after.empty else r
        verdict = agents['verifier'].verify(r, after_row, plan)
        print(f"   Closed: {'✅ Yes' if verdict['closed'] else '❌ No'}")
        
        # 8. Record
        print("8️⃣  Knowledge Curation...")
        case = {
            'cell': cell,
            'time': str(t0),
            'fingerprint': fp,
            'rca': rca,
            'plan': plan,
            'exec': exec_res,
            'verdict': verdict,
            'quality': cleaned_result['quality'],
            'detection': det
        }
        agents['curator'].record(case)
        results.append(case)
        print("   Learning extracted and stored")
        
        # Delay between cases
        if idx < len(candidates):
            time.sleep(1)
    
    return results

def save_artifacts(clf, baselines, cases):
    """Save learned artifacts"""
    print("\n💾 Saving artifacts...")
    
    import joblib
    joblib.dump(clf, 'mvp_detector.pkl')
    baselines.to_csv('mvp_baselines.csv')
    with open('mvp_cases.json','w') as f:
        json.dump(cases, f, indent=2)
    
    print('✅ Artifacts saved:')
    print('   - mvp_detector.pkl (anomaly detection model)')
    print('   - mvp_baselines.csv (cell baseline metrics)')
    print('   - mvp_cases.json (self-healing case history)')

def print_summary(cases):
    """Print summary of results"""
    print(f"\n{'='*70}")
    print("📊 SUMMARY")
    print(f"{'='*70}")
    
    closed = sum(1 for c in cases if c['verdict']['closed'])
    print(f"Total Cases: {len(cases)}")
    print(f"Closed: {closed}/{len(cases)} ({closed/len(cases)*100:.1f}%)")
    
    # Root cause breakdown
    causes = defaultdict(int)
    for c in cases:
        causes[c['rca']['primary_cause']] += 1
    
    print(f"\nRoot Cause Breakdown:")
    for cause, count in sorted(causes.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cause:20s}: {count:3d} cases")
    
    print(f"\n{'='*70}")

def main():
    """Main orchestrator"""
    print("="*70)
    print("🚀 SELF-HEALING RAN ORCHESTRATOR")
    print("="*70)
    
    # Load data
    df, feat_cols, baselines = load_and_prepare_data()
    
    # Train detector
    clf, explainer = train_detector(df, feat_cols)
    
    # Initialize agents
    agents = initialize_agents(feat_cols, baselines, clf, explainer)
    
    # Process cases
    results = process_cases(df, agents, num_cases=5)
    
    # Save artifacts
    save_artifacts(clf, baselines, agents['curator'].cases)
    
    # Print summary
    print_summary(results)
    
    print("\n✅ Orchestration complete!")

if __name__ == "__main__":
    main()
