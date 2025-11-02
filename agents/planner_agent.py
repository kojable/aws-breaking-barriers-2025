"""
Planner Agent - AWS Bedrock Implementation
Remediation Planning with action whitelist validation
"""
import json
import time
import random
from typing import Dict, Any
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
class PlannerAgent:
    api_key: str = None
    region_name: str = "us-east-1"
    model_id: str = "us.amazon.nova-lite-v1:0"
    
    ACTION_WHITELIST = {
        'scheduler_protection': {
            'params': ['cell', 'on'],
            'reversible': True,
            'risk': 'low',
            'description': 'Enable/disable scheduler protection for interference mitigation'
        },
        'tilt_neighbor': {
            'params': ['neighbor', 'delta_db'],
            'reversible': True,
            'risk': 'low',
            'description': 'Adjust antenna tilt of neighbor cell'
        },
        'enable_load_balancing': {
            'params': ['cluster', 'cio_delta_db'],
            'reversible': True,
            'risk': 'low',
            'description': 'Enable load balancing with CIO adjustment'
        },
        'raise_prb_cap': {
            'params': ['cell', 'delta'],
            'reversible': True,
            'risk': 'low',
            'description': 'Increase PRB capacity limit'
        },
        'restart_signaling_stack': {
            'params': ['cell'],
            'reversible': True,
            'risk': 'medium',
            'description': 'Restart signaling stack'
        },
        'check_backhaul': {
            'params': ['cell'],
            'reversible': False,
            'risk': 'low',
            'description': 'Verify backhaul connectivity'
        },
        'limit_tx_power': {
            'params': ['cell', 'preset'],
            'reversible': True,
            'risk': 'medium',
            'description': 'Limit transmission power'
        },
        'soft_reset_rru': {
            'params': ['cell'],
            'reversible': True,
            'risk': 'medium',
            'description': 'Soft reset of Remote Radio Unit'
        },
        'adjust_scheduler_weights': {
            'params': ['cell', 'profile'],
            'reversible': True,
            'risk': 'low',
            'description': 'Adjust scheduler weights'
        },
        'reduce_tx_power': {
            'params': ['cell', 'delta_db'],
            'reversible': True,
            'risk': 'low',
            'description': 'Reduce transmission power'
        },
        'handover_parameters': {
            'params': ['cell', 'threshold_delta'],
            'reversible': True,
            'risk': 'low',
            'description': 'Adjust handover thresholds'
        }
    }
    
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
                print(f"⚠️  PlannerAgent: Could not initialize Bedrock client: {e}")
                self.bedrock_runtime = None
        else:
            self.bedrock_runtime = None

    def _invoke_claude(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        """Invoke model via AWS Bedrock with retry logic"""
        if not self.bedrock_runtime:
            raise ValueError("Bedrock runtime client not available")
        
        body = json.dumps({
            "messages": [{
                "role": "user",
                "content": [{"text": prompt}]
            }],
            "inferenceConfig": {
                "max_new_tokens": max_tokens,
                "temperature": temperature
            }
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
                        print(f"⏳ Throttled (Planner). Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                raise ValueError(f"Bedrock API error: {e}")
        raise ValueError("Max retries exceeded")
    
    def plan(self, rca: Dict[str, Any], cell: str) -> Dict[str, Any]:
        """Generate remediation plan using LLM (LLM-only, no fallback)"""
        if not self.bedrock_runtime or not HAVE_BEDROCK:
            raise ValueError("PlannerAgent requires AWS Bedrock - LLM-only mode")
        
        return self._llm_plan(rca, cell)
    
    def _llm_plan(self, rca: Dict[str, Any], cell: str) -> Dict[str, Any]:
        """Use LLM to generate intelligent remediation plan"""
        rca_summary = {
            'primary_cause': rca.get('primary_cause', 'unknown'),
            'confidence': rca.get('confidence', 0.0),
            'evidence': rca.get('evidence', [])[:5],
            'secondary_hypotheses': rca.get('secondary', [])[:2]
        }
        
        available_actions = {
            action_id: {
                'description': info['description'],
                'params': info['params'],
                'reversible': info['reversible'],
                'risk': info['risk']
            }
            for action_id, info in self.ACTION_WHITELIST.items()
        }
        
        prompt = f"""Create a remediation plan for this RAN network issue.

Cell: {cell}

RCA Diagnosis:
{json.dumps(rca_summary, indent=2)}

Available Actions (WHITELIST - only use these):
{json.dumps(available_actions, indent=2)}

Propose 1-3 actions from the whitelist, ordered by increasing risk.

Return ONLY valid JSON:
{{
  "actions": [
    {{
      "id": "<action_id_from_whitelist>",
      "params": {{<required_params>}},
      "reversible": <true|false>,
      "rationale": "<explanation>"
    }}
  ],
  "prechecks": [<list of conditions>],
  "postchecks": [<list of success criteria>],
  "sla": {{
    "verify_minutes": <number>,
    "success_criteria": "<description>"
  }}
}}"""

        result_text = self._invoke_claude(prompt, max_tokens=2048, temperature=0.3).strip()
        
        if '```json' in result_text:
            import re
            match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)
        
        llm_plan = json.loads(result_text)
        
        # Validate actions against whitelist
        validated_actions = []
        for action in llm_plan.get('actions', []):
            action_id = action.get('id')
            if action_id in self.ACTION_WHITELIST:
                action['reversible'] = self.ACTION_WHITELIST[action_id]['reversible']
                validated_actions.append(action)
            else:
                print(f"⚠️  Action '{action_id}' not in whitelist, skipping")
        
        if not validated_actions:
            raise ValueError(f"LLM proposed no valid actions. RCA: {rca['primary_cause']}")
        
        return {
            'actions': validated_actions,
            'prechecks': llm_plan.get('prechecks', ["no critical alarms", "within change window"]),
            'postchecks': llm_plan.get('postchecks', []),
            'sla': llm_plan.get('sla', {"verify_minutes": 15, "success_criteria": "all_postchecks_green"})
        }
