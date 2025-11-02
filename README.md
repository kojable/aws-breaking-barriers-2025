# 🚀 Self-Healing RAN with AWS Bedrock - Complete Guide

## Overview

This is a production-ready **Self-Healing RAN (Radio Access Network)** system using **AWS Bedrock** and **8 specialized AI agents**. The system automatically detects network anomalies, diagnoses root causes, plans remediation actions, executes them safely, and learns from every case.

## 🎯 What This System Does

1. **Ingests** network KPI data and validates quality
2. **Detects** anomalies using ML + LLM insights
3. **Profiles** anomaly patterns with intelligent signal selection
4. **Diagnoses** root causes with multi-hypothesis reasoning
5. **Plans** safe, reversible remediation actions
6. **Executes** actions with safety guardrails
7. **Verifies** success through post-checks
8. **Learns** from every case for future recommendations

## 📋 Prerequisites

### Required
- Python 3.9 or higher
- AWS Account with Bedrock access
- AWS CLI configured with credentials
- Input data file: `cell_kpi_anomalies.csv`

### AWS Bedrock Models
- **Amazon Nova Lite** (`us.amazon.nova-lite-v1:0`) - Default, fast and cost-effective
- OR **Claude 4.5 Sonnet** - More powerful

## 🚀 Quick Start

### 1. Setup Environment

```cmd
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install boto3 botocore pandas numpy scikit-learn python-dotenv pydantic pydantic-settings joblib

# Optional: Install SHAP for ML explainability
pip install shap
```

### 2. Configure AWS

```cmd
# Configure AWS credentials
aws configure

# Verify access
aws sts get-caller-identity

# Check Bedrock model access
aws bedrock list-foundation-models --region us-east-1
```

**Important**: Request model access in [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/):
- Navigate to **Model access** → **Request model access**
- Enable **Amazon Nova Lite** (or Claude 3.5 Sonnet)
- Wait for approval (usually instant)

### 3. Run the System

```cmd
# Run orchestrator
python orchestrator.py
```

That's it! The system will:
- Load and prepare data
- Train the anomaly detection model
- Initialize all 8 agents
- Process anomaly cases
- Save results and learned artifacts

## 📁 Project Structure

```
aws-breaking-barrier/
│
├── orchestrator.py                    # Main workflow orchestrator
├── bedrock_config.py                  # Configuration module
├── cell_kpi_anomalies.csv            # Input data
│
├── agents/                            # All 8 agent implementations
│   ├── __init__.py
│   ├── base_agent.py                 # Base class (optional)
│   ├── ingestion_agent.py            # Data quality validation
│   ├── detector_agent.py             # Anomaly detection
│   ├── pattern_profiler_agent.py     # Signal fingerprinting
│   ├── rca_agent.py                  # Root cause analysis
│   ├── planner_agent.py              # Remediation planning
│   ├── executor_agent.py             # Safe execution
│   ├── verifier_agent.py             # Post-check validation
│   └── knowledge_curator.py          # Learning extraction
│
├── results/                           # Output directory
├── logs/                              # Execution logs
└── .cache/                            # Cached data
```

## 🤖 The 8 Agents

### 1. IngestionQualityAgent
**Purpose**: Data Quality Validation  
**LLM**: Optional (has fallback)

- Checks data completeness and missing values
- Detects outliers and anomalies
- Validates data types and ranges
- Reports quality metrics

### 2. DetectorAgent
**Purpose**: Anomaly Detection  
**LLM**: Optional (has fallback)

- Runs supervised ML model (Random Forest)
- Predicts anomaly probability
- Provides LLM-powered severity assessment
- Identifies likely network issues

### 3. PatternProfilerAgent
**Purpose**: Anomaly Fingerprinting  
**LLM**: Optional (has fallback)

- Generates compact fingerprint of KPI deviations
- Selects top 8 most significant signals
- Optionally adds SHAP attribution
- Intelligent signal selection via LLM

**Key Metrics Tracked**:
- `DRB.UEBlerDl` - Block Error Rate
- `DRB.UECqiDl` - Channel Quality Indicator
- `DRB.UEThpDl` - Throughput (downlink)
- `RRU.PrbTotDl` - PRB Utilization
- `RRC_SuccRate` - RRC Success Rate
- `Viavi.PEE.EnergyEfficiency` - Energy Efficiency
- `PEE.AvgPower` - Average Power

### 4. RCAAagent
**Purpose**: Root Cause Analysis  
**LLM**: Optional (has fallback)

- Analyzes fingerprint to determine root cause
- Provides confidence score and evidence
- Proposes secondary hypotheses
- Multi-hypothesis reasoning via LLM

**Root Cause Categories**:
- `interference` - High BLER + Low CQI
- `high_load` - High PRB + Low throughput
- `connection_failure` - Low RRC success rate
- `power_anomaly` - Energy efficiency issues
- `low_throughput` - Throughput degraded
- `inconclusive` - Insufficient evidence

### 5. PlannerAgent
**Purpose**: Remediation Planning  
**LLM**: **REQUIRED** (NO fallback)

- Proposes 1-3 remediation actions from whitelist
- Orders actions by increasing risk
- Provides rationale for each action
- Defines prechecks and postchecks
- Validates actions against whitelist

**Action Whitelist** (11 actions):
1. `scheduler_protection` - Enable/disable scheduler protection
2. `tilt_neighbor` - Adjust antenna tilt
3. `enable_load_balancing` - Enable load balancing with CIO
4. `raise_prb_cap` - Increase PRB capacity
5. `restart_signaling_stack` - Restart signaling (medium risk)
6. `check_backhaul` - Verify backhaul connectivity
7. `limit_tx_power` - Limit transmission power
8. `soft_reset_rru` - Soft reset RRU (medium risk)
9. `adjust_scheduler_weights` - Optimize scheduler
10. `reduce_tx_power` - Reduce TX power
11. `handover_parameters` - Adjust handover thresholds

### 6. ExecutorAgent
**Purpose**: Safe Execution with Guardrails  
**LLM**: Optional (has fallback)

- Validates all prechecks
- Checks blast radius (≤3 cells)
- Verifies change window compliance
- Executes actions through APIs
- Auto-rollback on failure

**Safety Guardrails**:
- Max blast radius: 3 cells
- Change window: Required for production
- Auto-rollback: On postcheck failure
- Manual intervention: For medium+ risk actions

### 7. VerifierAgent
**Purpose**: Post-check Evaluation  
**LLM**: **REQUIRED** (NO fallback)

- Compares AFTER vs BEFORE KPIs
- Compares AFTER vs BASELINE
- Evaluates each postcheck from plan
- Determines case closure
- Assesses residual risk

**Evaluation Criteria**:
- ✅ Lower BLER is better
- ✅ Higher CQI is better
- ✅ Higher Throughput is better
- ✅ Higher RRC Success Rate is better
- ✅ Higher Energy Efficiency is better

### 8. KnowledgeCurator
**Purpose**: Learning Extraction & Case Similarity  
**LLM**: Optional (has fallback)

- Records every case for learning
- Extracts structured patterns
- Identifies success/failure factors
- Enables case similarity matching
- Provides citations for future cases

## ⚙️ Configuration

### Model Selection

Edit `orchestrator.py`:

```python
# Option 1: Amazon Nova Lite (default - fast, cost-effective)
MODEL_ID = "us.amazon.nova-lite-v1:0"

# Option 2: Claude 3.5 Sonnet (most capable)
MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

# Option 3: Claude 3 Haiku (balanced)
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
```

### Enable/Disable LLM per Agent

```python
USE_LLM_FOR = {
    'ingestion': True,      # Quality assessment (has fallback)
    'detection': True,      # Anomaly insights (has fallback) 
    'profiling': True,      # Signal selection (has fallback)
    'rca': True,            # Root cause analysis (has fallback)
    'planning': True,       # ⚠️ NO FALLBACK - Must be True!
    'execution': True,      # Safe execution (has fallback)
    'verification': True,   # ⚠️ NO FALLBACK - Must be True!
    'curation': True,       # Learning extraction (has fallback)
}
```

**Note**: `planning` and `verification` agents **require LLM** - they have no fallback.

### Number of Cases to Process

```python
# In orchestrator.py, change num_cases parameter
results = process_cases(df, agents, num_cases=5)  # Change to 30 for full run
```

## 📊 API Usage Per Case

When all 8 agents use LLM:

| Agent | API Calls | Can Disable? |
|-------|-----------|--------------|
| IngestionQualityAgent | 1 | ✅ Yes (fallback available) |
| DetectorAgent | 1 | ✅ Yes (ML still works) |
| PatternProfilerAgent | 1 | ✅ Yes (rule-based selection) |
| RCAAagent | 1 | ✅ Yes (rule-based RCA) |
| PlannerAgent | 1 | ❌ NO (LLM required!) |
| ExecutorAgent | 1 | ✅ Yes (simulated mode) |
| VerifierAgent | 1 | ❌ NO (LLM required!) |
| KnowledgeCurator | 1 | ✅ Yes (storage only) |
| **TOTAL** | **8 calls/case** | Min 2 required |

⚠️ **Minimum 2 API calls per case** required (Planner + Verifier have NO fallback)

## 💰 Cost Estimation

### Amazon Nova Lite Pricing (us-east-1)
- **Input**: $0.06 per 1M tokens
- **Output**: $0.24 per 1M tokens

### Estimated Costs
- **Development/Testing (5 cases)**: < $0.10
- **Production (1000 cases/day)**: $10-30/month

### Cost Optimization Tips
1. Use Nova Lite for simple tasks
2. Disable optional LLM agents during testing
3. Batch process when possible
4. Use smaller max_tokens when appropriate

## 📈 Output Files

After running, you'll get:

1. **mvp_detector.pkl** - Trained anomaly detection model
2. **mvp_baselines.csv** - Cell baseline metrics
3. **mvp_cases.json** - Complete case history with:
   - Fingerprints
   - RCA diagnoses
   - Remediation plans
   - Execution results
   - Verification verdicts
   - Learned patterns

## 🔧 Troubleshooting

### Issue: `ThrottlingException` from Bedrock
**Solution**: Already handled with exponential backoff. If persistent, increase retry delay or request quota increase.

### Issue: `AccessDeniedException`
**Solution**:
1. Check IAM permissions
2. Verify model access in Bedrock Console
3. Confirm region is us-east-1

### Issue: Agent import errors
**Solution**:
1. Verify `agents/__init__.py` exists
2. Check Python path
3. Reinstall dependencies

### Issue: Slow performance
**Solution**:
1. Use Nova Lite instead of Claude
2. Reduce number of test cases
3. Disable optional LLM agents
4. Process in batches

## 🎓 Next Steps

### Phase 1: Basic Testing (Day 1)
- ✅ Run with 5 cases
- ✅ Verify all agents work
- ✅ Check output files

### Phase 2: Full Testing (Week 1)
- Run with 30+ cases
- Analyze closure rates
- Review learned patterns
- Tune confidence thresholds

### Phase 3: Production (Week 2-3)
- Deploy to AWS Lambda
- Set up CI/CD pipeline
- Configure auto-scaling
- Implement alerting

### Phase 4: Optimization (Week 4)
- Performance tuning
- Cost optimization
- Advanced features
- Documentation

## 📚 Additional Resources

### AWS Documentation
- [Bedrock User Guide](https://docs.aws.amazon.com/bedrock/)
- [Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)

### Model Documentation
- [Amazon Nova Models](https://aws.amazon.com/bedrock/nova/)
- [Claude 3.5 Sonnet](https://www.anthropic.com/claude)

## 🎉 Summary

You now have a complete, production-ready self-healing RAN system!

**Key Benefits:**
- ✅ Production-ready architecture
- ✅ Modular, maintainable code
- ✅ Scalable to handle large workloads
- ✅ Integrated with AWS services
- ✅ Comprehensive error handling
- ✅ Full observability
- ✅ Learning from every case

**Quick Commands:**
```cmd
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install boto3 pandas numpy scikit-learn joblib

# Configure AWS
aws configure

# Run
python orchestrator.py
```

---

**Document Version**: 1.0  
**Last Updated**: November 1, 2025  
**Estimated Setup Time**: 30 minutes  
**Estimated First Run**: 2 minutes
