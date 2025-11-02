"""
Enhanced Verification Script for Amazon Bedrock AgentCore Setup
Self-Healing RAN MVP with Strands Agents SDK

This script verifies:
1. Python version and dependencies
2. AWS credentials and Bedrock access
3. Directory structure
4. Configuration files
5. All 8 agent files
6. Data files
"""

import sys
import os
from pathlib import Path
import json

def check_python_version():
    """Check if Python version is 3.9 or higher"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✅ PASS Python Version ({version.major}.{version.minor}.{version.micro})")
        return True
    else:
        print(f"❌ FAIL Python Version ({version.major}.{version.minor}.{version.micro})")
        print("   Required: Python 3.9 or higher")
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\nChecking Python dependencies...")
    required_packages = {
        'boto3': 'AWS SDK',
        'botocore': 'AWS Core',
        'pandas': 'Data Processing',
        'numpy': 'Numerical Computing',
        'sklearn': 'Machine Learning',
        'dotenv': 'Environment Variables',
        'pydantic': 'Configuration',
        'jupyter': 'Jupyter Notebook'
    }
    
    missing = []
    installed = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            installed.append(f"{package} ({description})")
        except ImportError:
            missing.append(f"{package} ({description})")
    
    if not missing:
        print("✅ PASS Python Dependencies")
        print(f"   Installed: {len(installed)}/{len(required_packages)} packages")
        return True
    else:
        print("❌ FAIL Python Dependencies")
        print(f"   Missing packages ({len(missing)}):")
        for pkg in missing:
            print(f"      - {pkg}")
        print("\n   Run: pip install -r requirements_bedrock.txt")
        return False

def check_aws_credentials():
    """Check if AWS credentials are configured"""
    print("\nChecking AWS credentials...")
    try:
        import boto3
        sts = boto3.client('sts', region_name='us-east-1')
        identity = sts.get_caller_identity()
        print("✅ PASS AWS Credentials")
        print(f"   Account: {identity['Account']}")
        print(f"   User: {identity['Arn'].split('/')[-1]}")
        print(f"   Region: us-east-1")
        return True
    except Exception as e:
        print("❌ FAIL AWS Credentials")
        print(f"   Error: {str(e)[:100]}")
        print("   Run: aws configure")
        return False

def check_bedrock_access():
    """Check if Bedrock service is accessible"""
    print("\nChecking Bedrock service access...")
    try:
        import boto3
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        models = bedrock.list_foundation_models()
        model_count = len(models.get('modelSummaries', []))
        print("✅ PASS Bedrock Service Access")
        print(f"   Available models: {model_count}")
        return True
    except Exception as e:
        print("❌ FAIL Bedrock Service Access")
        print(f"   Error: {str(e)[:100]}")
        print("   Check IAM permissions for bedrock:ListFoundationModels")
        return False

def check_bedrock_model():
    """Check if specific Bedrock models are accessible"""
    print("\nChecking Bedrock model access...")
    try:
        import boto3
        import json
        
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Test Nova Lite (primary model)
        try:
            body = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "test"}]
                    }
                ]
            })
            
            response = bedrock_runtime.invoke_model(
                modelId="us.amazon.nova-lite-v1:0",
                body=body
            )
            print("✅ PASS Bedrock Model Access (Amazon Nova Lite)")
            return True
        except Exception as nova_error:
            # Try Claude as fallback
            try:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 100,
                    "messages": [
                        {
                            "role": "user",
                            "content": "test"
                        }
                    ]
                })
                
                response = bedrock_runtime.invoke_model(
                    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                    body=body
                )
                print("✅ PASS Bedrock Model Access (Claude 3.5 Sonnet)")
                print("   Note: Nova Lite not available, using Claude")
                return True
            except Exception as claude_error:
                print("⚠️  WARNING Bedrock Model Access")
                print("   Neither Nova Lite nor Claude accessible")
                print("   You need to request model access:")
                print("   1. Go to: https://console.aws.amazon.com/bedrock/")
                print("   2. Click: Model access")
                print("   3. Enable: Amazon Nova Lite OR Anthropic Claude")
                return False
    except Exception as e:
        print("❌ FAIL Bedrock Model Access")
        print(f"   Error: {str(e)[:100]}")
        return False

def check_directory_structure():
    """Check if required directories exist"""
    print("\nChecking directory structure...")
    required_dirs = {
        'agents': 'Agent implementations',
        'results': 'Execution results',
        'logs': 'Log files',
        '.cache': 'Cached data',
        'tests': 'Test files (optional)'
    }
    
    missing = []
    existing = []
    
    for dir_name, description in required_dirs.items():
        if Path(dir_name).exists():
            existing.append(f"{dir_name}/ ({description})")
        else:
            missing.append(f"{dir_name}/ ({description})")
    
    if not missing:
        print("✅ PASS Directory Structure")
        print(f"   All {len(required_dirs)} directories exist")
        return True
    else:
        print("❌ FAIL Directory Structure")
        print(f"   Missing directories ({len(missing)}):")
        for dir_info in missing:
            print(f"      - {dir_info}")
        print("\n   Run: setup_complete.cmd")
        return False

def check_config_files():
    """Check if configuration files exist"""
    print("\nChecking configuration files...")
    required_files = {
        '.env': 'Environment variables',
        'bedrock_config.py': 'Configuration module',
        'requirements_bedrock.txt': 'Dependencies list'
    }
    
    missing = []
    existing = []
    
    for file_name, description in required_files.items():
        if Path(file_name).exists():
            existing.append(f"{file_name} ({description})")
        else:
            missing.append(f"{file_name} ({description})")
    
    if not missing:
        print("✅ PASS Configuration Files")
        print(f"   All {len(required_files)} config files exist")
        return True
    else:
        print("⚠️  WARNING Configuration Files")
        print(f"   Missing files ({len(missing)}):")
        for file_info in missing:
            print(f"      - {file_info}")
        return False

def check_agent_files():
    """Check if all 8 agent files exist"""
    print("\nChecking agent files...")
    required_agents = {
        'base_agent.py': 'Base agent class',
        'ingestion_agent.py': 'Data quality validation',
        'detector_agent.py': 'Anomaly detection',
        'pattern_profiler_agent.py': 'Signal fingerprinting',
        'rca_agent.py': 'Root cause analysis',
        'planner_agent.py': 'Remediation planning',
        'executor_agent.py': 'Action execution',
        'verifier_agent.py': 'Post-execution validation',
        'knowledge_curator.py': 'Learning extraction'
    }
    
    missing = []
    existing = []
    
    for agent_file, description in required_agents.items():
        agent_path = Path(f'agents/{agent_file}')
        if agent_path.exists():
            # Check if file has content (not empty)
            if agent_path.stat().st_size > 0:
                existing.append(f"{agent_file} ({description})")
            else:
                missing.append(f"{agent_file} ({description}) - EMPTY")
        else:
            missing.append(f"{agent_file} ({description})")
    
    if not missing:
        print("✅ PASS Agent Files")
        print(f"   All {len(required_agents)} agent files exist")
        return True
    else:
        print("⚠️  WARNING Agent Files")
        print(f"   Status: {len(existing)}/{len(required_agents)} agents ready")
        if missing:
            print(f"   Missing or empty ({len(missing)}):")
            for agent_info in missing:
                print(f"      - {agent_info}")
        print("\n   See: BEDROCK_AGENTCORE_MIGRATION_GUIDE.md")
        return False

def check_data_files():
    """Check if data files exist"""
    print("\nChecking data files...")
    data_file = Path('cell_kpi_anomalies.csv')
    
    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print("✅ PASS Data Files")
        print(f"   Found: {data_file.name} ({size_mb:.2f} MB)")
        return True
    else:
        print("⚠️  WARNING Data Files")
        print(f"   Missing: {data_file.name}")
        print("   This file is needed to run the workflow")
        return False

def check_strands_sdk():
    """Check if Strands SDK is installed"""
    print("\nChecking Strands SDK...")
    try:
        import strands
        print("✅ PASS Strands SDK")
        print("   Strands Agents SDK is installed")
        return True
    except ImportError:
        print("⚠️  INFO Strands SDK")
        print("   Strands SDK not installed (optional)")
        print("   Install with: pip install git+https://github.com/awslabs/strands.git")
        return True  # Not critical, return True

def print_summary(checks, check_names):
    """Print summary of all checks"""
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\nResults: {passed}/{total} checks passed\n")
    
    for i, (check_result, check_name) in enumerate(zip(checks, check_names), 1):
        status = "✅ PASS" if check_result else "❌ FAIL"
        print(f"{i}. {status} - {check_name}")
    
    print("\n" + "=" * 70)
    
    if passed == total:
        print("🎉 SUCCESS! All checks passed.")
        print("\nYou're ready to start developing!")
        print("\nNext steps:")
        print("1. Review: BEDROCK_AGENTCORE_MIGRATION_GUIDE.md")
        print("2. Create agent files (if not done)")
        print("3. Run: jupyter notebook")
        print("4. Open: lite_healing_bedrock_agentcore.ipynb")
        return 0
    elif passed >= total * 0.7:  # 70% or more passed
        print("⚠️  PARTIAL SUCCESS - Most checks passed")
        print("\nYou can proceed with development, but fix warnings when possible.")
        print("\nSee: BEDROCK_AGENTCORE_MIGRATION_GUIDE.md for help")
        return 0
    else:
        print("❌ SETUP INCOMPLETE - Critical issues found")
        print("\nPlease fix the failed checks before proceeding.")
        print("\nQuick fixes:")
        print("1. Run: setup_complete.cmd")
        print("2. Run: aws configure")
        print("3. Request Bedrock model access in AWS Console")
        return 1

def main():
    """Run all verification checks"""
    print("=" * 70)
    print("AWS Bedrock AgentCore Setup Verification")
    print("Self-Healing RAN MVP with Strands Agents SDK")
    print("=" * 70)
    print()
    
    check_names = [
        "Python Version",
        "Python Dependencies",
        "AWS Credentials",
        "Bedrock Service Access",
        "Bedrock Model Access",
        "Directory Structure",
        "Configuration Files",
        "Agent Files",
        "Data Files",
        "Strands SDK"
    ]
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_aws_credentials(),
        check_bedrock_access(),
        check_bedrock_model(),
        check_directory_structure(),
        check_config_files(),
        check_agent_files(),
        check_data_files(),
        check_strands_sdk()
    ]
    
    return print_summary(checks, check_names)

if __name__ == "__main__":
    sys.exit(main())
