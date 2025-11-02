"""
Pattern Profiler Agent - AWS Bedrock Implementation
Generates compact fingerprints of KPI deviations
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List
from dataclasses import dataclass, field

try:
    import boto3
    from botocore.config import Config
    HAVE_BEDROCK = True
except ImportError:
    HAVE_BEDROCK = False

BEDROCK_CONFIG = Config(
    retries={'max_attempts': 1, 'mode': 'standard'},
    read_timeout=120,
    connect_timeout=10
)

@dataclass
class PatternProfilerAgent:
    baselines: pd.DataFrame
    feat_cols: List[str] = field(default_factory=list)
    shap_explainer: Any = None
    api_key: str = None
    region_name: str = "us-east-1"
    model_id: str = "us.amazon.nova-lite-v1:0"

    def __post_init__(self):
        """Initialize Bedrock client"""
        if HAVE_BEDROCK:
            try:
                self.bedrock_runtime = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=self.region_name,
                    config=BEDROCK_CONFIG
                )
            except Exception as e:
                print(f"⚠️  PatternProfilerAgent: Could not initialize Bedrock client: {e}")
                self.bedrock_runtime = None
        else:
            self.bedrock_runtime = None

    def _invoke_claude(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        """Invoke model via AWS Bedrock"""
        if not self.bedrock_runtime:
            raise ValueError("Bedrock runtime client not available")
        
        body = json.dumps({
            "messages": [{
                "role": "user",
                "content": [{"text": prompt}]
            }]
        })
        
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_id,
            body=body
        )
        response_body = json.loads(response['body'].read())
        return response_body['output']['message']['content'][0]['text']

    def fingerprint(self, row: pd.Series) -> List[Dict[str, Any]]:
        """Generate compact fingerprint of key KPI deviations"""
        cell = row['cell_name']
        metrics = ['DRB.UEBlerDl','DRB.UECqiDl','DRB.UEThpDl','RRU.PrbTotDl','RRC_SuccRate','Viavi.PEE.EnergyEfficiency','PEE.AvgPower']
        all_deltas = []
        
        for m in metrics:
            if m in row and cell in self.baselines.index and m in self.baselines.columns:
                base = self.baselines.loc[cell, m]
                cur = row[m]
                if pd.isna(base) or base == 0:
                    delta = np.nan
                else:
                    delta = (cur - base) / (base + 1e-9) * 100
                
                if not pd.isna(delta):
                    direction = 'up' if delta >= 0 else 'down'
                    all_deltas.append({
                        'metric': m, 
                        'value': float(cur), 
                        'baseline': float(base), 
                        'delta_pct': float(delta), 
                        'dir': direction
                    })
        
        if self.bedrock_runtime and HAVE_BEDROCK and len(all_deltas) > 0:
            try:
                fp = self._llm_select_signals(all_deltas, row)
            except Exception as e:
                print(f"⚠️  LLM profiling failed, using simple fallback")
                fp = self._simple_select_signals(all_deltas)
        else:
            fp = self._simple_select_signals(all_deltas)
        
        # Add SHAP attribution if available
        if self.shap_explainer is not None and self.feat_cols:
            try:
                X_row = pd.DataFrame([row[self.feat_cols].fillna(0.0).to_dict()])
                shap_vals = self.shap_explainer.shap_values(X_row)
                if isinstance(shap_vals, list) and len(shap_vals) > 1:
                    sv = shap_vals[1][0]
                else:
                    sv = shap_vals[0] if hasattr(shap_vals, 'shape') and len(shap_vals.shape) > 1 else shap_vals
                    if hasattr(sv, '__len__') and len(sv) > 0:
                        sv = sv[0] if hasattr(sv[0], '__len__') else sv
                shap_map = dict(zip(self.feat_cols, sv))
                top_shap = sorted(shap_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
                for feat, val in top_shap:
                    fp.append({'metric': feat, 'shap_value': float(val), 'shap_dir': 'positive' if val>0 else 'negative', 'type': 'shap'})
            except Exception:
                pass
        
        return fp
    
    def _simple_select_signals(self, all_deltas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple fallback: select top 8 signals by absolute delta"""
        return sorted(all_deltas, key=lambda d: abs(d['delta_pct']), reverse=True)[:8]
    
    def _llm_select_signals(self, all_deltas: List[Dict[str, Any]], row: pd.Series) -> List[Dict[str, Any]]:
        """Use LLM to intelligently select most relevant signals"""
        deltas_summary = [{
            'metric': item['metric'],
            'current': item['value'],
            'baseline': item['baseline'],
            'delta_pct': item['delta_pct'],
            'direction': item['dir']
        } for item in all_deltas]
        
        prompt = f"""Select the most significant ≤8 KPI signals that best characterize this anomaly pattern.

Available KPI Deltas:
{json.dumps(deltas_summary, indent=2)}

Return ONLY valid JSON array:
[
  {{
    "metric": "<metric_name>",
    "value": <current_value>,
    "baseline": <baseline_value>,
    "delta_pct": <delta_percentage>,
    "dir": "<up|down>"
  }}
]"""

        result_text = self._invoke_claude(prompt, max_tokens=1024, temperature=0.1).strip()
        
        if '```json' in result_text:
            import re
            match = re.search(r'```json\s*(\[.*?\])\s*```', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)
        
        llm_signals = json.loads(result_text)
        validated_signals = []
        for sig in llm_signals[:8]:
            if all(k in sig for k in ['metric', 'value', 'baseline', 'delta_pct', 'dir']):
                validated_signals.append({
                    'metric': sig['metric'],
                    'value': float(sig['value']),
                    'baseline': float(sig['baseline']),
                    'delta_pct': float(sig['delta_pct']),
                    'dir': sig['dir']
                })
        
        return validated_signals if validated_signals else self._simple_select_signals(all_deltas)
