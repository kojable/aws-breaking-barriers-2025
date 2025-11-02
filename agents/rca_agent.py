"""
RCA Agent - AWS Bedrock Implementation
Root Cause Analysis with multi-hypothesis reasoning
"""
import json
import time
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from dataclasses import dataclass

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
class RCAAagent:
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
                print(f"⚠️  RCAAagent: Could not initialize Bedrock client: {e}")
                self.bedrock_runtime = None
        else:
            self.bedrock_runtime = None

    def _invoke_claude(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
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
                        print(f"⏳ Throttled (RCA). Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                raise ValueError(f"Bedrock API error: {e}")
        raise ValueError("Max retries exceeded")
    
    def diagnose(self, fingerprint: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Root Cause Analysis using LLM or rule-based fallback"""
        if self.bedrock_runtime and HAVE_BEDROCK:
            try:
                return self._llm_diagnose(fingerprint)
            except Exception as e:
                print(f"⚠️  LLM RCA failed, using rule-based fallback: {e}")
                return self._rule_based_diagnose(fingerprint)
        else:
            return self._rule_based_diagnose(fingerprint)
    
    def _rule_based_diagnose(self, fingerprint: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple rule-based RCA (fallback)"""
        m = {x['metric']: x for x in fingerprint if x.get('type') != 'shap'}
        def d(metric, default=0):
            return m.get(metric, {}).get('delta_pct', default)

        evidence = []
        cause = 'inconclusive'
        conf = 0.5

        bler_up = d('DRB.UEBlerDl') > 30
        cqi_down = d('DRB.UECqiDl') < -20
        prb_up = d('RRU.PrbTotDl') > 25
        thp_down = d('DRB.UEThpDl') < -20
        rrc_down = d('RRC_SuccRate') < -10
        eff_down = d('Viavi.PEE.EnergyEfficiency') < -20
        power_up = d('PEE.AvgPower') > 10

        if bler_up and cqi_down and not prb_up:
            cause = 'interference'
            conf = 0.8
            evidence += [f"BLER up {d('DRB.UEBlerDl'):.1f}%", f"CQI down {d('DRB.UECqiDl'):.1f}%", "PRB not high"]
        elif prb_up and thp_down:
            cause = 'high_load'
            conf = 0.75
            evidence += [f"DL PRB up {d('RRU.PrbTotDl'):.1f}%", f"Throughput down {d('DRB.UEThpDl'):.1f}%"]
        elif rrc_down:
            cause = 'connection_failure'
            conf = 0.7
            evidence += [f"RRC success rate down {d('RRC_SuccRate'):.1f}%"]
        elif eff_down or power_up:
            cause = 'power_anomaly'
            conf = 0.7
            evidence += [f"Energy efficiency down {d('Viavi.PEE.EnergyEfficiency'):.1f}%"]
        elif thp_down and not prb_up:
            cause = 'low_throughput'
            conf = 0.65
            evidence += [f"Throughput down {d('DRB.UEThpDl'):.1f}%", "Load not high"]

        shap_items = [x for x in fingerprint if x.get('type') == 'shap']
        if shap_items:
            evidence.append(f"SHAP top drivers: {', '.join([s['metric'] for s in shap_items[:3]])}")

        return {
            'primary_cause': cause,
            'confidence': max(0.0, min(1.0, conf)),
            'evidence': evidence,
            'secondary': []
        }
    
    def _llm_diagnose(self, fingerprint: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Use LLM for intelligent root cause analysis"""
        delta_signals = [x for x in fingerprint if x.get('type') != 'shap']
        shap_signals = [x for x in fingerprint if x.get('type') == 'shap']
        
        fingerprint_summary = {
            'delta_based_signals': [{
                'metric': sig['metric'],
                'delta_pct': sig['delta_pct'],
                'direction': sig['dir'],
                'current_value': sig['value'],
                'baseline': sig['baseline']
            } for sig in delta_signals],
            'shap_attribution': [{
                'feature': sig['metric'],
                'shap_value': sig['shap_value'],
                'direction': sig['shap_dir']
            } for sig in shap_signals] if shap_signals else None
        }
        
        prompt = f"""Analyze this RAN network fingerprint and provide root cause diagnosis.

Fingerprint Data:
{json.dumps(fingerprint_summary, indent=2)}

Known Root Cause Categories:
- interference: High BLER + Low CQI, typically without high PRB
- high_load: High PRB utilization + Low throughput
- connection_failure: Low RRC success rate
- power_anomaly: Low energy efficiency or abnormal power
- low_throughput: Low throughput without high load
- inconclusive: Insufficient or contradictory evidence

Return ONLY valid JSON:
{{
  "primary_cause": "<root_cause_type>",
  "confidence": <0.0-1.0>,
  "evidence": [<list of evidence strings>],
  "secondary": [
    {{
      "cause": "<root_cause_type>",
      "confidence": <0.0-1.0>,
      "evidence": [<list of evidence strings>]
    }}
  ]
}}"""

        result_text = self._invoke_claude(prompt, max_tokens=1024, temperature=0.2).strip()
        
        if '```json' in result_text:
            import re
            match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)
        
        llm_result = json.loads(result_text)
        
        valid_causes = ['interference', 'high_load', 'connection_failure', 'power_anomaly', 'low_throughput', 'inconclusive']
        primary_cause = llm_result.get('primary_cause', 'inconclusive')
        if primary_cause not in valid_causes:
            primary_cause = 'inconclusive'
        
        return {
            'primary_cause': primary_cause,
            'confidence': float(llm_result.get('confidence', 0.5)),
            'evidence': llm_result.get('evidence', []),
            'secondary': llm_result.get('secondary', [])
        }
