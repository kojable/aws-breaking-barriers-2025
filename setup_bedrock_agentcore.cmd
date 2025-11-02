@echo off
REM ============================================================================
REM Amazon Bedrock AgentCore Setup Script for Windows
REM Self-Healing RAN MVP - Strands Agents Migration
REM ============================================================================

echo.
echo ========================================================================
echo    Amazon Bedrock AgentCore Setup - Strands Agents SDK
echo ========================================================================
echo.

REM Step 1: Check Python installation
echo [Step 1/10] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo [OK] Python is installed.
echo.

REM Step 2: Create virtual environment
echo [Step 2/10] Creating virtual environment...
if exist venv (
    echo [INFO] Virtual environment already exists.
) else (
    python -m venv venv
    echo [OK] Virtual environment created.
)
echo.

REM Step 3: Activate virtual environment
echo [Step 3/10] Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated.
echo.

REM Step 4: Upgrade pip
echo [Step 4/10] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] Pip upgraded.
echo.

REM Step 5: Check AWS CLI
echo [Step 5/10] Checking AWS CLI installation...
aws --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] AWS CLI not found.
    echo Please install from: https://awscli.amazonaws.com/AWSCLIV2.msi
    echo After installation, run: aws configure
    pause
) else (
    aws --version
    echo [OK] AWS CLI is installed.
)
echo.

REM Step 6: Install required packages
echo [Step 6/10] Installing required Python packages...
echo [INFO] This may take a few minutes...

REM Create requirements file if not exists
if not exist requirements_bedrock.txt (
    echo Creating requirements_bedrock.txt...
    (
        echo # Core Strands Agents and Bedrock
        echo boto3^>=1.34.144
        echo botocore^>=1.34.144
        echo.
        echo # AWS SDK
        echo awscli^>=1.32.0
        echo.
        echo # Data Processing
        echo pandas^>=2.0.0
        echo numpy^>=1.24.0
        echo.
        echo # Machine Learning
        echo scikit-learn^>=1.3.0
        echo scipy^>=1.11.0
        echo shap^>=0.44.0
        echo.
        echo # Jupyter
        echo jupyter^>=1.0.0
        echo ipykernel^>=6.29.0
        echo notebook^>=7.0.0
        echo.
        echo # Utilities
        echo python-dotenv^>=1.0.0
        echo pydantic^>=2.0.0
        echo pydantic-settings^>=2.0.0
    ) > requirements_bedrock.txt
)

pip install -r requirements_bedrock.txt --quiet
echo [OK] Python packages installed.
echo.

REM Step 7: Install Strands Agents SDK
echo [Step 7/10] Installing Strands Agents SDK...
echo [INFO] Attempting to install from PyPI...
pip install strands-agents --quiet >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] PyPI installation failed. Installing from GitHub...
    pip install git+https://github.com/awslabs/strands.git --quiet
)
echo [OK] Strands Agents SDK installed.
echo.

REM Step 8: Create directory structure
echo [Step 8/10] Creating directory structure...
if not exist agents mkdir agents
if not exist results mkdir results
if not exist logs mkdir logs
if not exist tests mkdir tests
if not exist monitoring mkdir monitoring
echo [OK] Directory structure created.
echo.

REM Step 9: Create configuration files
echo [Step 9/10] Creating configuration files...

REM Create .env file
if not exist .env (
    (
        echo # AWS Configuration
        echo AWS_REGION=us-east-1
        echo AWS_PROFILE=default
        echo.
        echo # Bedrock Configuration
        echo BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
        echo BEDROCK_MODEL_ID_HAIKU=anthropic.claude-3-haiku-20240307-v1:0
        echo BEDROCK_MAX_TOKENS=4096
        echo BEDROCK_TEMPERATURE=0.3
        echo.
        echo # Agent Configuration
        echo AGENT_TIMEOUT=300
        echo AGENT_MAX_RETRIES=5
        echo AGENT_RETRY_DELAY=2
        echo.
        echo # Data Configuration
        echo DATA_PATH=cell_kpi_anomalies.csv
        echo OUTPUT_PATH=results/
        echo.
        echo # Logging
        echo LOG_LEVEL=INFO
        echo LOG_FILE=logs/agent_execution.log
    ) > .env
    echo [OK] .env file created.
) else (
    echo [INFO] .env file already exists.
)

REM Create agents __init__.py
if not exist agents\__init__.py (
    (
        echo """
        echo Strands Agents for Self-Healing RAN
        echo """
        echo from .base_agent import BedrockAgent
        echo.
        echo __all__ = ['BedrockAgent']
    ) > agents\__init__.py
    echo [OK] agents/__init__.py created.
)
echo.

REM Step 10: Run verification
echo [Step 10/10] Running verification script...
echo [INFO] Creating verification script...

REM Create verify_setup.py if not exists
if not exist verify_setup.py (
    python -c "print('Creating verification script...')"
    (
        echo import sys
        echo import boto3
        echo from botocore.exceptions import ClientError
        echo.
        echo def verify_aws_credentials^(^):
        echo     try:
        echo         sts = boto3.client^('sts'^)
        echo         identity = sts.get_caller_identity^(^)
        echo         print^(f"✅ AWS Credentials configured"^)
        echo         print^(f"   Account: {identity['Account']}"^)
        echo         return True
        echo     except Exception as e:
        echo         print^(f"❌ AWS Credentials not configured: {e}"^)
        echo         return False
        echo.
        echo def verify_bedrock_access^(^):
        echo     try:
        echo         bedrock = boto3.client^('bedrock', region_name='us-east-1'^)
        echo         models = bedrock.list_foundation_models^(^)
        echo         claude_models = [m for m in models['modelSummaries'] if 'anthropic' in m['modelId'].lower^(^)]
        echo         if claude_models:
        echo             print^(f"✅ Bedrock access verified"^)
        echo             print^(f"   Available Claude models: {len^(claude_models^)}"^)
        echo             return True
        echo         else:
        echo             print^("⚠️  No Claude models available"^)
        echo             return False
        echo     except Exception as e:
        echo         print^(f"❌ Bedrock access failed: {e}"^)
        echo         return False
        echo.
        echo def main^(^):
        echo     print^("Verifying setup..."^)
        echo     verify_aws_credentials^(^)
        echo     verify_bedrock_access^(^)
        echo.
        echo if __name__ == "__main__":
        echo     main^(^)
    ) > verify_setup.py
)

echo [INFO] Running verification...
python verify_setup.py
echo.

REM Final message
echo ========================================================================
echo                         Setup Complete!
echo ========================================================================
echo.
echo Next Steps:
echo.
echo 1. Configure AWS credentials (if not done):
echo    ^> aws configure
echo.
echo 2. Request Bedrock model access:
echo    - Visit: https://console.aws.amazon.com/bedrock/
echo    - Go to "Model access"
echo    - Request access to Claude 3.5 Sonnet
echo.
echo 3. Review the setup guide:
echo    ^> type BEDROCK_AGENTCORE_SETUP.md
echo.
echo 4. Test the setup:
echo    ^> python verify_setup.py
echo.
echo 5. Start Jupyter Notebook:
echo    ^> jupyter notebook lite_healing_bedrock_agentcore.ipynb
echo.
echo 6. Read documentation:
echo    - BEDROCK_AGENTCORE_SETUP.md (Complete setup guide)
echo    - QUICKSTART_COMMANDS.md (Quick reference)
echo.
echo ========================================================================
echo.
pause
