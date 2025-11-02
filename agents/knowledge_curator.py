"""
Knowledge Curator Agent - AWS Bedrock Implementation
Learning extraction and case similarity
"""
import json
from typing import Dict, Any, List
from dataclasses import dataclass, field

try:
    import boto3
    HAVE_BEDROCK = True
except ImportError:
    HAVE_BEDROCK = False

@dataclass
class KnowledgeCurator:
    api_key: str = None
    region_name: str = "us-east-1"
    model_id: str = "us.amazon.nova-lite-v1:0"
    cases: List[Dict[str, Any]] = field(default_factory=list)
    case_embeddings: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize Bedrock client"""
        if HAVE_BEDROCK:
            try:
                self.bedrock_runtime = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=self.region_name
                )
            except Exception as e:
                print(f"⚠️  KnowledgeCurator: Could not initialize Bedrock client: {e}")
                self.bedrock_runtime = None
        else:
            self.bedrock_runtime = None

    def _invoke_claude(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
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
    
    def record(self, case: Dict[str, Any]):
        """Record a case and extract learning using LLM"""
        self.cases.append(case)
        
        if self.bedrock_runtime and HAVE_BEDROCK:
            try:
                learning = self._extract_learning(case)
                self.case_embeddings.append(learning)
            except Exception as e:
                print(f"⚠️  KnowledgeCurator learning extraction failed: {e}")
    
    def _extract_learning(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to extract structured learning from a case"""
        case_summary = {
            'case_id': len(self.cases),
            'cell': case.get('cell', 'unknown'),
            'time': case.get('time', 'unknown'),
            'fingerprint': case.get('fingerprint', [])[:5],
            'rca': {
                'primary_cause': case.get('rca', {}).get('primary_cause', 'unknown'),
                'confidence': case.get('rca', {}).get('confidence', 0.0)
            },
            'plan': {
                'actions': [{'id': a.get('id'), 'params': a.get('params')} 
                           for a in case.get('plan', {}).get('actions', [])[:3]]
            },
            'execution': {
                'status': case.get('exec', {}).get('status', 'unknown'),
                'executed_count': len(case.get('exec', {}).get('executed', []))
            },
            'outcome': {
                'closed': case.get('verdict', {}).get('closed', False),
                'residual_risk': case.get('verdict', {}).get('residual_risk', 'unknown')
            }
        }
        
        prompt = f"""Extract learning from this self-healing case for future recommendations.

Case Summary:
{json.dumps(case_summary, indent=2)}

Return ONLY valid JSON:
{{
  "case_id": <case number>,
  "pattern": {{
    "fingerprint_signature": ["<key signal 1>", "<key signal 2>"],
    "cause_pattern": "<root cause type>",
    "context": "<relevant context>"
  }},
  "solution": {{
    "actions_taken": [
      {{
        "action": "<action_id>",
        "params": {{<key params>}},
        "risk": "<low|medium|high>"
      }}
    ],
    "outcome": "<success|partial|failed>",
    "success_factors": ["<what made it work or fail>"]
  }},
  "learning": {{
    "what_worked": "<specific actions and why>",
    "what_failed": "<specific issues encountered>",
    "confidence": <0.0-1.0>,
    "applicability": "<when this solution applies>"
  }},
  "citation": "<Brief quote for future recommendations>",
  "similarity_keys": ["<key1>", "<key2>", "<key3>"]
}}"""

        try:
            result_text = self._invoke_claude(prompt, max_tokens=1024, temperature=0.3).strip()
            
            if '```json' in result_text:
                import re
                match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
                if match:
                    result_text = match.group(1)
            
            learning = json.loads(result_text)
            
            required_fields = ['case_id', 'pattern', 'solution', 'learning', 'citation']
            if not all(field in learning for field in required_fields):
                raise ValueError(f"LLM learning missing required fields")
            
            return learning
        except Exception as e:
            print(f"⚠️  KnowledgeCurator LLM error: {e}")
            raise
    
    def find_similar_cases(self, fingerprint: List[Dict], rca_cause: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar past cases to provide recommendations"""
        if not self.case_embeddings:
            return []
        
        current_signals = {item['metric'] for item in fingerprint if 'metric' in item}
        
        similar = []
        for learning in self.case_embeddings:
            pattern_signals = set(learning.get('pattern', {}).get('fingerprint_signature', []))
            pattern_cause = learning.get('pattern', {}).get('cause_pattern', '')
            
            signal_overlap = len(current_signals & pattern_signals) / max(len(current_signals), 1)
            cause_match = 1.0 if pattern_cause == rca_cause else 0.3
            score = (signal_overlap * 0.6) + (cause_match * 0.4)
            
            if score > 0.3:
                similar.append({
                    'learning': learning,
                    'similarity_score': score
                })
        
        similar.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar[:top_k]
    
    def get_recommendations(self, fingerprint: List[Dict], rca_cause: str) -> List[str]:
        """Get actionable recommendations based on similar past cases"""
        similar_cases = self.find_similar_cases(fingerprint, rca_cause, top_k=3)
        
        recommendations = []
        for item in similar_cases:
            learning = item['learning']
            citation = learning.get('citation', '')
            confidence = learning.get('learning', {}).get('confidence', 0.0)
            
            if citation and confidence > 0.5:
                recommendations.append(f"{citation} (confidence: {confidence:.1%})")
        
        return recommendations
