@echo off
REM ============================================================================
REM AWS Bedrock AgentCore - Complete Automated Setup
REM Self-Healing RAN MVP Migration Script
REM ============================================================================

echo.
echo ========================================================================
echo AWS Bedrock AgentCore - Automated Setup
echo Self-Healing RAN MVP with Strands Agents SDK
echo ========================================================================
echo.

REM Check Python installation
echo [1/12] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python not found!
    echo    Please install Python 3.9 or higher from https://www.python.org/
    pause
    exit /b 1
)

python --version
echo ✅ Python found
echo.

REM Create virtual environment
echo [2/12] Creating virtual environment...
if exist .venv (
    echo ⚠️  Virtual environment already exists, skipping...
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)
echo.

REM Activate virtual environment
echo [3/12] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment activated
echo.

REM Upgrade pip
echo [4/12] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo ✅ Pip upgraded
echo.

REM Install core dependencies
echo [5/12] Installing core dependencies (boto3, AWS CLI)...
pip install boto3 botocore awscli --quiet
if errorlevel 1 (
    echo ❌ ERROR: Failed to install core dependencies
    pause
    exit /b 1
)
echo ✅ Core dependencies installed
echo.

REM Install configuration packages
echo [6/12] Installing configuration packages...
pip install python-dotenv pydantic pydantic-settings --quiet
echo ✅ Configuration packages installed
echo.

REM Install data science packages
echo [7/12] Installing data science packages...
pip install pandas numpy scikit-learn shap --quiet
echo ✅ Data science packages installed
echo.

REM Install Jupyter
echo [8/12] Installing Jupyter...
pip install jupyter notebook ipykernel ipywidgets --quiet
echo ✅ Jupyter installed
echo.

REM Install visualization packages
echo [9/12] Installing visualization packages...
pip install matplotlib seaborn plotly --quiet
echo ✅ Visualization packages installed
echo.

REM Install development tools
echo [10/12] Installing development tools...
pip install pytest pytest-asyncio black flake8 --quiet
echo ✅ Development tools installed
echo.

REM Install Strands SDK
echo [11/12] Installing Strands Agents SDK from GitHub...
echo    This may take a few minutes...
pip install git+https://github.com/awslabs/strands.git --quiet
if errorlevel 1 (
    echo ⚠️  WARNING: Strands SDK installation failed
    echo    This is optional - you can continue without it
    echo    Or install manually later with: pip install git+https://github.com/awslabs/strands.git
) else (
    echo ✅ Strands SDK installed
)
echo.

REM Create directory structure
echo [12/12] Creating directory structure...
if not exist agents mkdir agents
if not exist results mkdir results
if not exist logs mkdir logs
if not exist .cache mkdir .cache
if not exist tests mkdir tests

REM Create __init__.py for agents package
type nul > agents\__init__.py

echo ✅ Directory structure created
echo.

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env configuration file...
    (
        echo # AWS Configuration
        echo AWS_REGION=us-east-1
        echo AWS_PROFILE=default
        echo.
        echo # Bedrock Model Configuration
        echo BEDROCK_MODEL_ID=us.amazon.nova-lite-v1:0
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
        echo OUTPUT_PATH=results
        echo CACHE_PATH=.cache
        echo.
        echo # Logging
        echo LOG_LEVEL=INFO
        echo LOG_FILE=logs/agent_execution.log
    ) > .env
    echo ✅ .env file created
) else (
    echo ⚠️  .env file already exists, skipping...
)
echo.

REM Create .gitignore if it doesn't exist
if not exist .gitignore (
    echo Creating .gitignore file...
    (
        echo .venv/
        echo __pycache__/
        echo *.pyc
        echo .env
        echo .cache/
        echo logs/
        echo results/
        echo *.log
        echo .DS_Store
        echo Thumbs.db
        echo .pytest_cache/
        echo .coverage
        echo htmlcov/
    ) > .gitignore
    echo ✅ .gitignore created
)
echo.

echo ========================================================================
echo Setup Complete!
echo ========================================================================
echo.
echo Next Steps:
echo.
echo 1. Configure AWS credentials:
echo    aws configure
echo.
echo 2. Request Bedrock model access:
echo    - Go to: https://console.aws.amazon.com/bedrock/
echo    - Click: Model access
echo    - Enable: Amazon Nova Lite
echo.
echo 3. Verify setup:
echo    python verify_setup.py
echo.
echo 4. Create agent files (see BEDROCK_AGENTCORE_MIGRATION_GUIDE.md)
echo.
echo 5. Start developing:
echo    jupyter notebook
echo.
echo ========================================================================
echo.
echo Would you like to configure AWS now? (Y/N)
set /p CONFIGURE_AWS=
if /i "%CONFIGURE_AWS%"=="Y" (
    echo.
    echo Running AWS configuration...
    aws configure
)
echo.

echo Would you like to run verification now? (Y/N)
set /p RUN_VERIFY=
if /i "%RUN_VERIFY%"=="Y" (
    if exist verify_setup.py (
        echo.
        echo Running verification...
        python verify_setup.py
    ) else (
        echo ⚠️  verify_setup.py not found. Please create it first.
    )
)
echo.

echo ========================================================================
echo Setup script completed successfully!
echo ========================================================================
echo.
echo Your virtual environment is activated.
echo To activate it in future sessions, run: .venv\Scripts\activate
echo.
pause
