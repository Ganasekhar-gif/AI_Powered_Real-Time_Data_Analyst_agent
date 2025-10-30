# core/rag/finance_kb.py

from typing import List, Dict, Any
from core.rag.index import FaissIndex


class FinanceKnowledgeBase:
    """
    Finance-specific knowledge base that provides domain expertise
    to all agents in the financial data analytics workflow.
    """
    
    def __init__(self, faiss_index: FaissIndex):
        self.faiss = faiss_index
        self.load_finance_knowledge()
    
    def load_finance_knowledge(self):
        """Load comprehensive financial domain knowledge into FAISS index."""
        
        # Financial Data Quality Rules
        data_quality_rules = [
            "Financial data quality: Revenue missing values should be 0, not mean/median",
            "Balance sheet validation: Assets must equal Liabilities plus Equity",
            "Cash flow validation: Operating + Investing + Financing = Net Change in Cash",
            "Financial ratios: P/E = Market Cap / Net Income, ROE = Net Income / Shareholders Equity",
            "Outlier detection: Use IQR method for financial data, not Z-score (financial data is skewed)",
            "Missing values strategy: Use forward fill for time series, 0 for revenue, median for ratios",
            "Duplicates check: Look for duplicate transactions, not just duplicate rows",
            "Negative values: Remove negative values for positive-only financial metrics",
            "Currency consistency: Always check for currency consistency across data",
            "Date formatting: Use consistent date formats for time series financial analysis",
        ]
        
        # Financial Analysis Patterns
        analysis_patterns = [
            "Trend analysis: Calculate Year-over-Year (YoY) and Quarter-over-Quarter (QoQ) growth rates",
            "Financial ratios: Current ratio = Current Assets / Current Liabilities",
            "Profitability metrics: Gross margin = (Revenue - COGS) / Revenue",
            "Liquidity ratios: Quick ratio = (Current Assets - Inventory) / Current Liabilities",
            "Efficiency ratios: Asset turnover = Revenue / Total Assets",
            "Leverage ratios: Debt-to-Equity = Total Debt / Total Equity",
            "Market ratios: Price-to-Book = Market Price per Share / Book Value per Share",
            "Growth rates: CAGR = (Ending Value / Beginning Value)^(1/n) - 1",
            "Risk metrics: Beta measures systematic risk, Sharpe ratio = (Return - Risk-free rate) / Volatility",
            "Valuation: DCF = Sum of (Cash Flow / (1 + Discount Rate)^n)",
        ]
        
        # Data Cleaning and Preprocessing Rules
        cleaning_rules = [
            "Financial data cleaning: Remove negative values for positive-only metrics like revenue",
            "Currency conversion: Always check for currency consistency before analysis",
            "Date handling: Use consistent date formats for time series analysis",
            "Categorical encoding: Use ordinal encoding for financial ratings (AAA, AA, A, BBB, etc.)",
            "Missing data: Forward fill for time series, 0 for revenue, median for ratios",
            "Outlier treatment: Use IQR method for financial data, consider business context",
            "Data validation: Check for logical inconsistencies in financial statements",
            "Duplicate handling: Remove duplicate transactions, keep latest entries",
            "Data types: Ensure numeric columns are properly formatted for calculations",
            "Scale normalization: Use log transformation for highly skewed financial data",
        ]
        
        # Financial Visualization Best Practices
        viz_patterns = [
            "Financial charts: Use line charts for time series, bar charts for comparisons",
            "Ratio visualization: Create radar charts for multiple financial ratios",
            "Trend analysis: Use candlestick charts for stock price movements",
            "Distribution: Use histograms for financial data distributions",
            "Correlation: Use heatmaps for correlation matrices of financial metrics",
            "Risk visualization: Use scatter plots for risk-return analysis",
            "Performance: Use waterfall charts for P&L analysis",
            "Comparison: Use grouped bar charts for peer comparison",
            "Growth: Use stacked area charts for revenue growth analysis",
            "Forecasting: Use line charts with confidence intervals for predictions",
        ]
        
        # SQL Query Patterns for Financial Data
        sql_patterns = [
            "Financial SQL: Calculate YoY growth: (current_value - LAG(current_value, 12)) / LAG(current_value, 12) * 100",
            "Ratio calculations: SELECT revenue, net_income, revenue/net_income as profit_margin FROM income_statement",
            "Trend analysis: Use window functions for moving averages and growth rates",
            "Peer comparison: Use RANK() and PERCENT_RANK() for relative performance",
            "Time series: Use DATE_TRUNC for grouping by quarters and years",
            "Financial filtering: Filter by date ranges for quarterly and annual analysis",
            "Aggregation: Use SUM, AVG, MIN, MAX for financial metrics aggregation",
            "Joins: Join income statement, balance sheet, and cash flow statements",
            "Subqueries: Use for complex financial calculations and comparisons",
            "CTEs: Use Common Table Expressions for multi-step financial analysis",
        ]
        
        # Python Data Processing Patterns
        python_patterns = [
            "Financial data processing: Use pandas for time series analysis and financial calculations",
            "Data cleaning: df.fillna(method='ffill') for time series, df.fillna(0) for revenue",
            "Financial calculations: Calculate ratios using vectorized operations for performance",
            "Time series: Use pandas resample() for quarterly and annual aggregations",
            "Outlier detection: Use IQR method: Q1 - 1.5*IQR, Q3 + 1.5*IQR",
            "Data validation: Check for logical inconsistencies in financial statements",
            "Visualization: Use matplotlib and plotly for financial charts and dashboards",
            "Statistical analysis: Use scipy for financial statistical tests and analysis",
            "Data export: Export cleaned data to CSV, Excel, or database formats",
            "Performance: Use vectorized operations for large financial datasets",
        ]
        
        # Combine all knowledge
        all_knowledge = (
            data_quality_rules + 
            analysis_patterns + 
            cleaning_rules + 
            viz_patterns + 
            sql_patterns + 
            python_patterns
        )
        
        # Add to FAISS index
        knowledge_docs = [{"text": rule} for rule in all_knowledge]
        self.faiss.add(knowledge_docs)
        self.faiss.save()
        
        print(f"[INFO] Loaded {len(all_knowledge)} financial knowledge rules into FAISS index")
    
    def get_financial_context(self, query: str, k: int = 5) -> str:
        """
        Get relevant financial knowledge for a given query.
        
        Args:
            query: The user query or context
            k: Number of relevant knowledge items to retrieve
            
        Returns:
            Formatted string with relevant financial knowledge
        """
        hits = self.faiss.search(query, k=k)
        if not hits:
            return "No relevant financial knowledge found."
        
        context_parts = []
        for i, hit in enumerate(hits, 1):
            context_parts.append(f"{i}. {hit['text']} (relevance: {hit['score']:.2f})")
        
        return "\n".join(context_parts)
    
    def get_data_quality_rules(self, data_type: str = "general") -> str:
        """Get specific data quality rules for financial data."""
        quality_queries = {
            "income_statement": "revenue expenses income statement data quality",
            "balance_sheet": "balance sheet assets liabilities equity validation",
            "cash_flow": "cash flow statement operating investing financing",
            "market_data": "stock price volume market data quality",
            "general": "financial data quality validation rules"
        }
        
        query = quality_queries.get(data_type, quality_queries["general"])
        return self.get_financial_context(query, k=3)
    
    def get_analysis_patterns(self, analysis_type: str = "general") -> str:
        """Get specific analysis patterns for financial data."""
        analysis_queries = {
            "ratios": "financial ratios calculation P/E ROE current ratio",
            "trends": "trend analysis YoY QoQ growth rates",
            "risk": "risk analysis beta sharpe ratio volatility",
            "valuation": "valuation DCF P/E price-to-book",
            "general": "financial analysis patterns methods"
        }
        
        query = analysis_queries.get(analysis_type, analysis_queries["general"])
        return self.get_financial_context(query, k=3)
    
    def get_cleaning_suggestions(self, issue_type: str = "general") -> str:
        """Get specific cleaning suggestions for financial data issues."""
        cleaning_queries = {
            "missing_values": "missing values financial data cleaning forward fill",
            "outliers": "outliers financial data IQR detection",
            "duplicates": "duplicates financial transactions cleaning",
            "negative_values": "negative values financial metrics cleaning",
            "general": "financial data cleaning preprocessing rules"
        }
        
        query = cleaning_queries.get(issue_type, cleaning_queries["general"])
        return self.get_financial_context(query, k=3)
