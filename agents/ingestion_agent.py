"""
Ingestion Quality Agent - AWS Bedrock Implementation
Data quality validation and preprocessing
"""
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, Any, List
from dataclasses import dataclass

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config
    HAVE_BEDROCK = True
except ImportError:
    HAVE_BEDROCK = False
    boto3 = None

BEDROCK_CONFIG = Config(
    retries={'max_attempts': 1, 'mode': 'standard'},
    read_timeout=120,
    connect_timeout=10
)

@dataclass
class IngestionQualityAgent:
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
                print(f"⚠️  Could not initialize Bedrock client: {e}")
                self.bedrock_runtime = None
        else:
            self.bedrock_runtime = None

    def _invoke_claude(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
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

    def process(self, batch: pd.DataFrame) -> Dict[str, Any]:
        """Process batch using LLM for intelligent data quality assessment"""
        if not HAVE_BEDROCK or not self.bedrock_runtime:
            return self._simple_process(batch)
        
        try:
            batch_summary = {
                'total_rows': len(batch),
                'columns': self.feat_cols[:10],
                'sample_stats': {
                    col: {
                        'mean': float(batch[col].mean()) if col in batch.columns else None,
                        'missing_count': int(batch[col].isna().sum()) if col in batch.columns else None,
                        'missing_pct': float(batch[col].isna().mean() * 100) if col in batch.columns else None
                    }
                    for col in self.feat_cols[:5]
                }
            }
            
            prompt = f"""Analyze this RAN KPI data batch and provide quality assessment.

Batch Summary:
{json.dumps(batch_summary, indent=2)}

Return ONLY valid JSON:
{{
    "quality": {{
        "rows": <number>,
        "missing_pct": <overall missing percentage as float>,
        "issues": [<list of detected issues as strings>],
        "severity": "<low|medium|high>",
        "recommendations": [<list of recommendations>]
    }}
}}"""

            result_text = self._invoke_claude(prompt, max_tokens=512, temperature=0.1).strip()
            
            if '```json' in result_text:
                match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
                if match:
                    result_text = match.group(1)
            elif '```' in result_text:
                match = re.search(r'```\s*(\{.*?\})\s*```', result_text, re.DOTALL)
                if match:
                    result_text = match.group(1)
            
            quality_result = json.loads(result_text)
            batch_filled = batch.copy()
            batch_filled[self.feat_cols] = batch_filled[self.feat_cols].fillna(0.0)
            
            return {
                'clean': batch_filled,
                'quality': quality_result.get('quality', {
                    'rows': len(batch),
                    'missing_pct': 0.0,
                    'issues': ['LLM response format error'],
                    'severity': 'unknown'
                })
            }
        except Exception as e:
            print(f"⚠️  LLM processing failed ({str(e)[:80]}), using simple fallback")
            return self._simple_process(batch)
    
    def _simple_process(self, batch: pd.DataFrame) -> Dict[str, Any]:
        """Simple fallback quality check without LLM"""
        miss = batch[self.feat_cols].isna().mean().mean()
        out = {
            'rows': len(batch),
            'missing_pct': float(miss * 100),
            'issues': ['Basic check only - LLM unavailable'],
            'severity': 'low' if miss < 0.1 else 'medium' if miss < 0.3 else 'high'
        }
        batch_filled = batch.copy()
        batch_filled[self.feat_cols] = batch_filled[self.feat_cols].fillna(0.0)
        return {'clean': batch_filled, 'quality': out}
