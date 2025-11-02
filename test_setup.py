"""
Quick Setup Test Script
Verifies all dependencies and AWS configuration
"""
import sys
from pathlib import Path

def test_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (need 3.9+)")
        return False

def test_dependencies():
    """Check required packages"""
    required = ['boto3', 'pandas', 'numpy', 'sklearn', 'joblib']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing.append(package)
    
    return len(missing) == 0

def test_aws_credentials():
    """Check AWS credentials"""
    try:
        import boto3
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS credentials configured")
        print(f"   Account: {identity['Account']}")
        print(f"   User: {identity['Arn'].split('/')[-1]}")
        return True
    except Exception as e:
        print(f"❌ AWS credentials not configured: {str(e)[:50]}")
        return False

def test_bedrock_access():
    """Check Bedrock access"""
    try:
        import boto3
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        models = bedrock.list_foundation_models()
        print(f"✅ Bedrock access verified")
        print(f"   Available models: {len(models.get('modelSummaries', []))}")
        return True
    except Exception as e:
        print(f"❌ Bedrock access failed: {str(e)[:50]}")
        return False

def test_data_file():
    """Check input data file"""
    data_path = Path('cell_kpi_anomalies.csv')
    if data_path.exists():
        print(f"✅ Data file found: {data_path}")
        return True
    else:
        print(f"❌ Data file not found: {data_path}")
        return False

def test_agent_files():
    """Check agent files exist"""
    agents_dir = Path('agents')
    required_agents = [
        'ingestion_agent.py',
        'detector_agent.py',
        'pattern_profiler_agent.py',
        'rca_agent.py',
        'planner_agent.py',
        'executor_agent.py',
        'verifier_agent.py',
        'knowledge_curator.py'
    ]
    
    missing = []
    for agent in required_agents:
        agent_path = agents_dir / agent
        if agent_path.exists():
            print(f"✅ {agent}")
        else:
            print(f"❌ {agent} (missing)")
            missing.append(agent)
    
    return len(missing) == 0

def main():
    """Run all tests"""
    print("="*70)
    print("🔍 SETUP VERIFICATION")
    print("="*70)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("AWS Credentials", test_aws_credentials),
        ("Bedrock Access", test_bedrock_access),
        ("Data File", test_data_file),
        ("Agent Files", test_agent_files),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'─'*70}")
        print(f"Testing: {name}")
        print(f"{'─'*70}")
        results.append(test_func())
    
    print(f"\n{'='*70}")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 SUCCESS! All {total} checks passed.")
        print("\nYou're ready to run:")
        print("   python orchestrator.py")
    else:
        print(f"⚠️  {passed}/{total} checks passed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Configure AWS: aws configure")
        print("   - Request Bedrock access in AWS Console")
    
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
