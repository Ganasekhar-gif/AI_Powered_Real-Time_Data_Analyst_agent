# core/agents/data_examiner_agent.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from core.llm.groq_client import GroqClient
from core.rag.index import FaissIndex
from core.rag.finance_kb import FinanceKnowledgeBase


class DataExaminerAgent:
    """
    Agent specialized in examining financial datasets for quality issues,
    data problems, and providing finance-specific recommendations.
    """
    
    def __init__(self, llm: GroqClient, faiss_index: FaissIndex, finance_kb: FinanceKnowledgeBase):
        self.llm = llm
        self.faiss = faiss_index
        self.finance_kb = finance_kb
    
    def examine_dataset(self, df: pd.DataFrame, dataset_type: str = "financial") -> Dict[str, Any]:
        """
        Examine a financial dataset for quality issues and provide recommendations.
        
        Args:
            df: The dataset to examine
            dataset_type: Type of financial dataset (income_statement, balance_sheet, etc.)
            
        Returns:
            Dictionary containing examination results and recommendations
        """
        # Get basic dataset statistics
        basic_stats = self._get_basic_statistics(df)
        
        # Get financial-specific data quality rules
        finance_context = self.finance_kb.get_data_quality_rules(dataset_type)
        
        # Get previous examination patterns from memory
        memory_context = self.faiss.search("data examination financial", k=3)
        memory_text = "\n".join([hit['text'] for hit in memory_context])
        
        # Build comprehensive examination prompt
        prompt = self._build_examination_prompt(df, basic_stats, finance_context, memory_text, dataset_type)
        
        # Get LLM analysis
        analysis = self.llm.complete(prompt).strip()
        
        # Store examination results in RAG memory
        examination_record = f"Financial data examination for {dataset_type}:\nShape: {df.shape}\nColumns: {list(df.columns)}\nAnalysis: {analysis}"
        self.faiss.add([{"text": examination_record}])
        self.faiss.save()
        
        # Generate specific recommendations
        recommendations = self._generate_recommendations(df, analysis)
        
        return {
            "type": "examination",
            "dataset_type": dataset_type,
            "basic_stats": basic_stats,
            "analysis": analysis,
            "recommendations": recommendations,
            "issues_found": self._extract_issues(analysis),
            "next_steps": self._suggest_next_steps(df, analysis)
        }
    
    def _get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistical information about the dataset."""
        stats = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicates": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        # Add financial-specific statistics
        if stats["numeric_columns"]:
            numeric_df = df[stats["numeric_columns"]]
            stats["numeric_summary"] = {
                "mean": numeric_df.mean().to_dict(),
                "median": numeric_df.median().to_dict(),
                "std": numeric_df.std().to_dict(),
                "min": numeric_df.min().to_dict(),
                "max": numeric_df.max().to_dict(),
                "negative_values": (numeric_df < 0).sum().to_dict(),
                "zero_values": (numeric_df == 0).sum().to_dict()
            }
        
        return stats
    
    def _build_examination_prompt(self, df: pd.DataFrame, basic_stats: Dict, 
                                finance_context: str, memory_context: str, dataset_type: str) -> str:
        """Build a comprehensive prompt for dataset examination."""
        
        # Sample data for analysis
        sample_data = df.head(3).to_string() if len(df) > 0 else "Empty dataset"
        
        return f"""
You are a Financial Data Quality Expert. Examine this {dataset_type} dataset for issues and provide detailed analysis.

FINANCIAL DATA QUALITY RULES:
{finance_context}

PREVIOUS EXAMINATION PATTERNS:
{memory_context}

DATASET INFORMATION:
- Shape: {basic_stats['shape']}
- Columns: {basic_stats['columns']}
- Data Types: {basic_stats['dtypes']}
- Missing Values: {basic_stats['missing_values']}
- Missing Percentage: {basic_stats['missing_percentage']}
- Duplicates: {basic_stats['duplicates']}
- Numeric Columns: {basic_stats['numeric_columns']}

SAMPLE DATA:
{sample_data}

NUMERIC SUMMARY:
{basic_stats.get('numeric_summary', 'No numeric columns')}

Please analyze this financial dataset and identify:

1. DATA QUALITY ISSUES:
   - Missing values and their financial implications
   - Duplicates and their business impact
   - Data type inconsistencies
   - Outliers and their significance
   - Negative values where they shouldn't exist
   - Zero values and their context

2. FINANCIAL VALIDATION ISSUES:
   - Balance sheet equation violations (Assets = Liabilities + Equity)
   - Cash flow statement inconsistencies
   - Revenue/expense logic errors
   - Ratio calculation problems
   - Time series continuity issues

3. DATA COMPLETENESS:
   - Missing critical financial metrics
   - Incomplete time periods
   - Missing categorical values
   - Inconsistent naming conventions

4. BUSINESS LOGIC ISSUES:
   - Unrealistic values (negative revenue, etc.)
   - Inconsistent units or scales
   - Missing context or metadata
   - Data integrity problems

Provide specific, actionable recommendations for each issue found.
"""
    
    def _generate_recommendations(self, df: pd.DataFrame, analysis: str) -> List[Dict[str, str]]:
        """Generate specific recommendations based on the analysis."""
        recommendations = []
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            for col in missing_cols:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                if missing_pct > 50:
                    recommendations.append({
                        "issue": f"High missing values in {col} ({missing_pct:.1f}%)",
                        "recommendation": "Consider dropping this column or investigate data source",
                        "priority": "High"
                    })
                elif missing_pct > 10:
                    recommendations.append({
                        "issue": f"Moderate missing values in {col} ({missing_pct:.1f}%)",
                        "recommendation": "Use appropriate imputation method (forward fill for time series, 0 for revenue)",
                        "priority": "Medium"
                    })
        
        # Check for duplicates
        if df.duplicated().sum() > 0:
            recommendations.append({
                "issue": f"Found {df.duplicated().sum()} duplicate rows",
                "recommendation": "Remove duplicates, keeping the most recent entries",
                "priority": "High"
            })
        
        # Check for negative values in positive-only columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.lower() in ['revenue', 'sales', 'income', 'profit', 'assets']:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    recommendations.append({
                        "issue": f"Negative values in {col} ({negative_count} instances)",
                        "recommendation": "Investigate and correct negative values",
                        "priority": "High"
                    })
        
        # Check for outliers
        for col in numeric_cols:
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > 0:
                    recommendations.append({
                        "issue": f"Potential outliers in {col} ({outliers} instances)",
                        "recommendation": "Review outliers for business context and data accuracy",
                        "priority": "Medium"
                    })
        
        return recommendations
    
    def _extract_issues(self, analysis: str) -> List[str]:
        """Extract specific issues from the analysis text."""
        # Simple keyword-based extraction (can be enhanced with NLP)
        issues = []
        
        issue_keywords = [
            "missing values", "duplicates", "outliers", "negative values",
            "data type", "inconsistent", "incomplete", "unrealistic",
            "validation", "integrity", "logic error"
        ]
        
        analysis_lower = analysis.lower()
        for keyword in issue_keywords:
            if keyword in analysis_lower:
                issues.append(keyword)
        
        return list(set(issues))
    
    def _suggest_next_steps(self, df: pd.DataFrame, analysis: str) -> List[str]:
        """Suggest next steps based on the analysis."""
        next_steps = []
        
        # Check for missing values
        if df.isnull().any().any():
            next_steps.append("Handle missing values using appropriate imputation methods")
        
        # Check for duplicates
        if df.duplicated().sum() > 0:
            next_steps.append("Remove duplicate rows")
        
        # Check for outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        has_outliers = False
        for col in numeric_cols:
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).any():
                    has_outliers = True
                    break
        
        if has_outliers:
            next_steps.append("Analyze and handle outliers")
        
        # General next steps
        next_steps.extend([
            "Perform exploratory data analysis",
            "Calculate financial ratios and metrics",
            "Create visualizations for data insights",
            "Validate data against business rules"
        ])
        
        return next_steps
    
    def get_financial_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get financial-specific insights from the dataset."""
        insights = {
            "financial_metrics": {},
            "data_quality_score": 0,
            "recommendations": []
        }
        
        # Calculate data quality score
        total_issues = 0
        max_issues = len(df.columns) * 3  # Missing, duplicates, outliers
        
        # Missing values penalty
        missing_penalty = df.isnull().sum().sum()
        total_issues += missing_penalty
        
        # Duplicates penalty
        duplicate_penalty = df.duplicated().sum()
        total_issues += duplicate_penalty
        
        # Outliers penalty
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_penalty = 0
        for col in numeric_cols:
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_penalty += outliers
        
        total_issues += outlier_penalty
        
        # Calculate quality score (0-100)
        insights["data_quality_score"] = max(0, 100 - (total_issues / max_issues * 100))
        
        return insights
