"""
Ingestion Quality Agent - AWS Bedrock Implementation
Data quality validation and preprocessing
"""
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, Any, List
from dataclasses import dataclass

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config
    HAVE_BEDROCK = True
except ImportError:
    HAVE_BEDROCK = False
    boto3 = None

BEDROCK_CONFIG = Config(
    retries={'max_attempts': 1, 'mode': 'standard'},
    read_timeout=120,
    connect_timeout=10
)

@dataclass
class IngestionQualityAgent:
    """
    Agent responsible for data quality checks and preprocessing.
    
    Capabilities:
    - Data validation and quality assessment
    - Missing value detection and handling
    - Outlier identification
    - Data type validation
    - Timestamp consistency checks
    """
    
    def __init__(self):
        super().__init__(
            name="IngestionQualityAgent",
            description="Validates data quality, checks for missing values, outliers, and ensures data consistency",
            instructions="""
You are a data quality specialist for telecom network KPI data.

Your responsibilities:
1. Check for missing values and data completeness
2. Identify outliers and anomalies in the data structure
3. Validate data types and ranges
4. Ensure timestamp consistency
5. Report data quality metrics
6. Suggest data cleaning strategies

When analyzing data:
- Be thorough and systematic
- Provide specific metrics and percentages
- Identify critical issues that could affect downstream analysis
- Suggest remediation strategies for data quality issues

Always provide detailed reports on data quality issues found.
            """
        )
        
        # Register tools
        self.register_tool(self.check_data_quality, "check_data_quality")
        self.register_tool(self.analyze_missing_values, "analyze_missing_values")
        self.register_tool(self.detect_outliers, "detect_outliers")
    
    async def execute(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute complete data quality analysis.
        
        Args:
            input_data: DataFrame with network KPI data
        
        Returns:
            Dictionary with quality report and cleaned data
        """
        self.logger.info(f"Starting ingestion quality check for {len(input_data)} rows")
        
        # Step 1: Basic quality checks
        quality_report = self.check_data_quality(input_data)
        
        # Step 2: Analyze missing values
        missing_analysis = self.analyze_missing_values(input_data)
        
        # Step 3: Detect outliers
        outlier_analysis = self.detect_outliers(input_data)
        
        # Step 4: Get AI-powered assessment
        ai_assessment = await self._get_ai_assessment(
            quality_report,
            missing_analysis,
            outlier_analysis
        )
        
        # Step 5: Clean data
        cleaned_data = self._clean_data(input_data, quality_report)
        
        self.logger.info("Ingestion quality check completed")
        
        return {
            'status': 'success',
            'agent': self.name,
            'quality_report': quality_report,
            'missing_analysis': missing_analysis,
            'outlier_analysis': outlier_analysis,
            'ai_assessment': ai_assessment,
            'cleaned_data': cleaned_data,
            'summary': {
                'total_rows': len(input_data),
                'clean_rows': len(cleaned_data),
                'rows_removed': len(input_data) - len(cleaned_data),
                'quality_score': quality_report['quality_score']
            }
        }
    
    @tool
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks.
        
        Args:
            df: Input DataFrame with KPI data
        
        Returns:
            Dictionary with quality metrics and issues
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicates': int(df.duplicated().sum()),
            'timestamp_issues': [],
            'data_types': {},
            'column_stats': {}
        }
        
        # Check missing values per column
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                quality_report['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(df) * 100)
                }
        
        # Check data types
        quality_report['data_types'] = {
            col: str(dtype) for col, dtype in df.dtypes.items()
        }
        
        # Calculate quality score
        missing_penalty = sum(v['percentage'] for v in quality_report['missing_values'].values()) / len(df.columns) if df.columns.size > 0 else 0
        duplicate_penalty = (quality_report['duplicates'] / len(df) * 100) if len(df) > 0 else 0
        quality_score = max(0, 100 - missing_penalty - duplicate_penalty)
        quality_report['quality_score'] = round(quality_score, 2)
        
        return quality_report
    
    @tool
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detailed analysis of missing value patterns.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary with missing value analysis
        """
        missing_analysis = {
            'total_missing': int(df.isna().sum().sum()),
            'missing_percentage': float(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100),
            'columns_with_missing': [],
            'rows_with_missing': int((df.isna().any(axis=1)).sum()),
            'complete_rows': int((~df.isna().any(axis=1)).sum())
        }
        
        # Analyze each column with missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_analysis['columns_with_missing'].append({
                    'column': col,
                    'missing_count': int(missing_count),
                    'missing_percentage': float(missing_count / len(df) * 100),
                    'data_type': str(df[col].dtype)
                })
        
        return missing_analysis
    
    @tool
    def detect_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers in numeric columns using IQR and Z-score methods.
        
        Args:
            df: Input DataFrame
            threshold: Z-score threshold for outlier detection
        
        Returns:
            Dictionary with outlier analysis
        """
        outlier_analysis = {
            'method': 'IQR and Z-score',
            'threshold': threshold,
            'columns_analyzed': [],
            'total_outliers': 0
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() == 0:
                continue
            
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            # Z-score method
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                z_outliers = (z_scores > threshold).sum()
            else:
                z_outliers = 0
            
            if iqr_outliers > 0 or z_outliers > 0:
                outlier_analysis['columns_analyzed'].append({
                    'column': col,
                    'iqr_outliers': int(iqr_outliers),
                    'z_score_outliers': int(z_outliers),
                    'percentage': float((iqr_outliers / len(df)) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'mean': float(mean),
                    'std': float(std)
                })
                
                outlier_analysis['total_outliers'] += int(iqr_outliers)
        
        return outlier_analysis
    
    async def _get_ai_assessment(
        self,
        quality_report: Dict,
        missing_analysis: Dict,
        outlier_analysis: Dict
    ) -> str:
        """
        Get AI-powered assessment of data quality issues.
        
        Args:
            quality_report: Quality metrics
            missing_analysis: Missing value analysis
            outlier_analysis: Outlier detection results
        
        Returns:
            AI-generated assessment and recommendations
        """
        prompt = f"""
Analyze this network KPI data quality report and provide a comprehensive assessment:

QUALITY METRICS:
- Total rows: {quality_report['total_rows']}
- Quality score: {quality_report['quality_score']}/100
- Duplicate rows: {quality_report['duplicates']}

MISSING VALUES:
- Total missing: {missing_analysis['total_missing']}
- Missing percentage: {missing_analysis['missing_percentage']:.2f}%
- Columns with missing data: {len(missing_analysis['columns_with_missing'])}

OUTLIERS:
- Total outliers detected: {outlier_analysis['total_outliers']}
- Columns with outliers: {len(outlier_analysis['columns_analyzed'])}

Please provide:
1. Overall data quality assessment (Good/Fair/Poor)
2. Critical issues that need immediate attention
3. Impact on downstream anomaly detection and analysis
4. Recommended data cleaning strategies
5. Risk assessment for proceeding with this data

Be specific and actionable.
"""
        
        assessment = self.invoke_model(prompt)
        return assessment
    
    def _clean_data(self, df: pd.DataFrame, quality_report: Dict) -> pd.DataFrame:
        """
        Clean data based on quality report.
        
        Args:
            df: Input DataFrame
            quality_report: Quality assessment results
        
        Returns:
            Cleaned DataFrame
        """
        cleaned = df.copy()
        
        # Remove duplicates
        if quality_report['duplicates'] > 0:
            cleaned = cleaned.drop_duplicates()
            self.logger.info(f"Removed {quality_report['duplicates']} duplicate rows")
        
        # Remove rows with excessive missing values (>50% of columns)
        threshold = len(cleaned.columns) * 0.5
        before_count = len(cleaned)
        cleaned = cleaned.dropna(thresh=len(cleaned.columns) - threshold)
        removed = before_count - len(cleaned)
        if removed > 0:
            self.logger.info(f"Removed {removed} rows with excessive missing values")
        
        return cleaned


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_ingestion_agent():
        """Test the ingestion agent"""
        # Create sample data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
            'cell_id': ['cell_001'] * 100,
            'throughput': np.random.normal(100, 10, 100),
            'latency': np.random.normal(50, 5, 100),
            'packet_loss': np.random.uniform(0, 1, 100)
        })
        
        # Add some missing values and duplicates
        sample_data.loc[10:15, 'throughput'] = np.nan
        sample_data = pd.concat([sample_data, sample_data.iloc[:5]], ignore_index=True)
        
        # Initialize and run agent
        agent = IngestionQualityAgent()
        result = await agent.execute(sample_data)
        
        # Print results
        print("\n" + "=" * 70)
        print("INGESTION QUALITY AGENT RESULTS")
        print("=" * 70)
        print(f"\nStatus: {result['status']}")
        print(f"Quality Score: {result['summary']['quality_score']}/100")
        print(f"Total Rows: {result['summary']['total_rows']}")
        print(f"Clean Rows: {result['summary']['clean_rows']}")
        print(f"Rows Removed: {result['summary']['rows_removed']}")
        
        print("\nAI Assessment:")
        print("-" * 70)
        print(result['ai_assessment'])
        print("=" * 70)
    
    # Run test
    asyncio.run(test_ingestion_agent())
