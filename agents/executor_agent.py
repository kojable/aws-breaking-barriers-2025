"""
Executor Agent - AWS Bedrock Implementation
Safe execution with guardrails
"""
import json
from typing import Dict, Any
from dataclasses import dataclass

try:
    import boto3
    HAVE_BEDROCK = True
except ImportError:
    HAVE_BEDROCK = False

@dataclass
class ExecutorAgent:
    api_key: str = None
    region_name: str = "us-east-1"
    model_id: str = "us.amazon.nova-lite-v1:0"
    
    GUARDRAILS = {
        'max_blast_radius': 3,
        'require_change_window': True,
        'auto_rollback_on_failure': True,
        'manual_intervention_threshold': 'medium',
    }
    
    def __post_init__(self):
        """Initialize Bedrock client"""
        if HAVE_BEDROCK:
            try:
                self.bedrock_runtime = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=self.region_name
                )
            except Exception as e:
                print(f"⚠️  ExecutorAgent: Could not initialize Bedrock client: {e}")
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
    
    def apply(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the remediation plan with safety guardrails"""
        if self.bedrock_runtime and HAVE_BEDROCK:
            try:
                return self._llm_execute(plan)
            except Exception as e:
                print(f"⚠️  LLM execution failed: {str(e)[:100]}")
                return self._simple_execute(plan)
        else:
            return self._simple_execute(plan)
    
    def _simple_execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Simple simulated execution (fallback)"""
        executed = []
        for a in plan['actions']:
            executed.append({
                "id": a['id'], 
                "status": "applied", 
                "params": a['params'],
                "method": "simulated"
            })
        return {
            "executed": executed, 
            "status": "ok",
            "mode": "simulated"
        }
    
    def _llm_execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to intelligently execute plan with guardrails"""
        execution_context = {
            'actions': plan.get('actions', []),
            'prechecks': plan.get('prechecks', []),
            'postchecks': plan.get('postchecks', []),
            'sla': plan.get('sla', {}),
            'guardrails': self.GUARDRAILS
        }
        
        affected_cells = set()
        for action in plan.get('actions', []):
            if 'cell' in action.get('params', {}):
                affected_cells.add(action['params']['cell'])
            if 'neighbor' in action.get('params', {}):
                neighbor = action['params']['neighbor']
                if neighbor != 'AUTO':
                    affected_cells.add(neighbor)
        
        prompt = f"""Execute this remediation plan safely with guardrails.

Execution Context:
{json.dumps(execution_context, indent=2)}

Affected Cells: {len(affected_cells)} cell(s)
Blast Radius Check: {'✅ PASS' if len(affected_cells) <= self.GUARDRAILS['max_blast_radius'] else '❌ FAIL'}

Return ONLY valid JSON:
{{
  "executed": [
    {{
      "id": "<action_id>",
      "status": "<applied|refused|deferred|rolled_back>",
      "params": {{<action_params>}},
      "method": "<NetConf|Ansible|OSS-API|ticket|simulated>",
      "reason": "<explanation>"
    }}
  ],
  "status": "<ok|partial|failed>",
  "mode": "<production|simulated>",
  "safety_checks": {{
    "blast_radius_ok": <true|false>,
    "prechecks_passed": <true|false>,
    "change_window_ok": <true|false>
  }}
}}"""

        result_text = self._invoke_claude(prompt, max_tokens=2048, temperature=0.2).strip()
        
        if '```json' in result_text:
            import re
            match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)
        
        return json.loads(result_text)
