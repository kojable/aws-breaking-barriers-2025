"""
Verifier Agent - AWS Bedrock Implementation
Post-check evaluation (LLM-only, no fallback)
"""
import json
import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass

try:
    import boto3
    HAVE_BEDROCK = True
except ImportError:
    HAVE_BEDROCK = False

@dataclass
class VerifierAgent:
    baselines: pd.DataFrame
    api_key: str = None
    region_name: str = "us-east-1"
    model_id: str = "us.amazon.nova-lite-v1:0"
    
    def __post_init__(self):
        """Initialize Bedrock client"""
        if HAVE_BEDROCK:
            try:
                self.bedrock_runtime = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=self.region_name
                )
            except Exception as e:
                print(f"⚠️  VerifierAgent: Could not initialize Bedrock client: {e}")
                self.bedrock_runtime = None
        else:
            self.bedrock_runtime = None

    def _invoke_claude(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.2) -> str:
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
    
    def verify(self, before_row: pd.Series, after_row: pd.Series, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Verify postchecks using LLM (LLM-only, no fallback)"""
        if not self.bedrock_runtime or not HAVE_BEDROCK:
            raise ValueError("VerifierAgent requires AWS Bedrock - LLM-only mode")
        
        return self._llm_verify(before_row, after_row, plan)
    
    def _llm_verify(self, before_row: pd.Series, after_row: pd.Series, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to evaluate postchecks"""
        cell = before_row.get('cell_name', 'unknown')
        key_metrics = ['DRB.UEBlerDl', 'DRB.UECqiDl', 'DRB.UEThpDl', 'RRU.PrbTotDl', 'RRC_SuccRate', 'Viavi.PEE.EnergyEfficiency']
        
        before_metrics = {m: float(before_row.get(m, 0)) for m in key_metrics if m in before_row.index}
        after_metrics = {m: float(after_row.get(m, 0)) for m in key_metrics if m in after_row.index}
        
        baseline_metrics = {}
        if cell in self.baselines.index:
            baseline_metrics = {m: float(self.baselines.loc[cell, m]) for m in key_metrics if m in self.baselines.columns}
        
        verification_context = {
            'cell': cell,
            'before': before_metrics,
            'after': after_metrics,
            'baseline': baseline_metrics,
            'postchecks': plan.get('postchecks', []),
            'actions_taken': [{'id': a.get('id'), 'params': a.get('params')} for a in plan.get('actions', [])]
        }
        
        prompt = f"""Evaluate if the remediation was successful by comparing BEFORE vs AFTER KPIs.

Verification Context:
{json.dumps(verification_context, indent=2)}

Evaluation Criteria:
- Lower BLER is better
- Higher CQI is better
- Higher Throughput is better
- Higher RRC Success Rate is better
- Higher Energy Efficiency is better

Return ONLY valid JSON:
{{
  "closed": <true|false>,
  "checks": [
    {{
      "postcheck": "<postcheck_condition>",
      "ok": <true|false>,
      "evidence": "<metric comparison>"
    }}
  ],
  "residual_risk": "<none|low|medium|high>",
  "kpi_improvements": {{
    "<metric>": {{
      "before": <value>,
      "after": <value>,
      "change_pct": <percentage>,
      "improved": <true|false>
    }}
  }},
  "overall_assessment": "<brief summary>"
}}"""

        try:
            result_text = self._invoke_claude(prompt, max_tokens=2048, temperature=0.2).strip()
            
            if '```json' in result_text:
                import re
                match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
                if match:
                    result_text = match.group(1)
            
            result = json.loads(result_text)
            
            if 'closed' not in result or 'checks' not in result:
                raise ValueError(f"LLM verification missing required fields")
            
            result['closed'] = bool(result['closed'])
            result['mode'] = 'llm'
            
            return result
        except Exception as e:
            print(f"⚠️  VerifierAgent LLM error: {e}")
            raise ValueError(f"VerifierAgent LLM verification failed: {e}")
