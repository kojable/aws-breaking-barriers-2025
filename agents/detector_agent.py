"""
Detector Agent - AWS Bedrock Implementation
Anomaly Detection with ML + LLM Insights
"""
import pandas as pd
import numpy as np
import json
import time
import random
from typing import Dict, Any, List
from dataclasses import dataclass, field

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config
    HAVE_BEDROCK = True
except ImportError:
    HAVE_BEDROCK = False
    boto3 = None

# Bedrock configuration
BEDROCK_CONFIG = Config(
    retries={'max_attempts': 1, 'mode': 'standard'},
    read_timeout=120,
    connect_timeout=10
)

@dataclass
class DetectorAgent:
    model: Any
    feat_cols: List[str]
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
                print(f"⚠️  DetectorAgent: Could not initialize Bedrock client: {e}")
                self.bedrock_runtime = None
        else:
            self.bedrock_runtime = None

    def _invoke_claude(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Invoke model via AWS Bedrock with retry logic"""
        if not self.bedrock_runtime:
            raise ValueError("Bedrock runtime client not available")
        
        body = json.dumps({
            "messages": [{
                "role": "user",
                "content": [{"text": prompt}]
            }]
        })
        
        max_retries = 5
        base_delay = 2
        max_delay = 60
        
        for attempt in range(max_retries):
            try:
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.model_id,
                    body=body
                )
                response_body = json.loads(response['body'].read())
                return response_body['output']['message']['content'][0]['text']
            except Exception as e:
                error_str = str(e)
                if 'ThrottlingException' in error_str or 'Too many requests' in error_str:
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                        print(f"⏳ Throttled (Detector). Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                raise ValueError(f"Bedrock API error: {e}")
        raise ValueError("Max retries exceeded")

    def detect(self, batch: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using ML + optional LLM insights"""
        X = batch[self.feat_cols].fillna(0.0)
        pred = self.model.predict(X)
        proba = self.model.predict_proba(X)[:, 1]
        
        result = {
            'is_anomaly_any': bool(pred.sum() > 0),
            'anomaly_count': int(pred.sum()),
            'max_probability': float(proba.max()),
            'predictions': pred.tolist(),
            'probabilities': proba.tolist()
        }
        
        if self.bedrock_runtime and HAVE_BEDROCK and result['is_anomaly_any']:
            try:
                insights = self._llm_insights(batch, pred, proba)
                result['llm_insights'] = insights
            except Exception as e:
                result['llm_insights'] = {'status': 'failed', 'error': str(e)[:100]}
        
        return result
    
    def _llm_insights(self, batch: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
        """Get LLM insights about detected anomalies"""
        anomaly_indices = np.where(predictions == 1)[0]
        if len(anomaly_indices) == 0:
            return {'status': 'no_anomalies'}
        
        sample_indices = anomaly_indices[:3]
        anomaly_samples = []
        key_metrics = ['DRB.UEBlerDl', 'DRB.UECqiDl', 'DRB.UEThpDl', 'RRU.PrbTotDl', 'RRC_SuccRate']
        
        for idx in sample_indices:
            row = batch.iloc[idx]
            sample = {
                'probability': float(probabilities[idx]),
                'metrics': {m: float(row.get(m, 0)) for m in key_metrics if m in row.index}
            }
            anomaly_samples.append(sample)
        
        prompt = f"""Analyze these RAN network anomalies and provide insights.

Detected Anomalies:
{json.dumps({'total_anomalies': len(anomaly_indices), 'samples': anomaly_samples}, indent=2)}

Return ONLY valid JSON:
{{
    "severity": "<low|medium|high|critical>",
    "likely_issues": [<list of potential network issues>],
    "recommendation": "<brief action recommendation>"
}}"""

        try:
            result_text = self._invoke_claude(prompt, max_tokens=512, temperature=0.2).strip()
            
            if '```json' in result_text:
                import re
                match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
                if match:
                    result_text = match.group(1)
            
            insights = json.loads(result_text)
            insights['status'] = 'success'
            return insights
        except Exception as e:
            return {'status': 'error', 'message': str(e)[:100]}
