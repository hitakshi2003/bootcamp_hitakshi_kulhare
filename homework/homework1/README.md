# Predicting Credit Risk for Small Business Loan Applications
**Stage:** Problem Framing & Scoping (Stage 01)

## Problem Statement
Small and medium enterprises (SMEs) are critical drivers of economic growth but often struggle to access financing due to perceived repayment risks. Traditional credit scoring methods rely heavily on manual assessments and outdated scoring models, leading to inefficiencies, inconsistent loan decisions, and missed opportunities for creditworthy applicants. This project aims to build a credit risk prediction model using both traditional financial data and other sources like payment history, transactions, and industry trends. The model aims to generate accurate, real-time risk scores, enabling lenders to make faster, more consistent, and data-driven loan approval decisions.

## Stakeholder & User
Primary Stakeholder: Commercial banks and financial institutions providing SME loans.  
End Users: Credit risk analysts and loan officers using the model to evaluate applications.  
Workflow Context: Model output will integrate into loan decision systems, delivering instant risk scores during the application review process.

## Useful Answer & Decision
Predictive — generating probability scores for loan default risk.  
Metric: Predicted default probability (0–1) and associated risk classification (Low/Medium/High).  
Artifact: A trained machine learning model with an API or dashboard for credit risk scoring.

## Assumptions & Constraints
- Historical SME loan data is available and representative of current lending conditions.
- Data sources include both financial statements and alternative data streams.
- Compliance with banking regulations and data privacy laws (e.g., GDPR, CCPA).
- Model latency should be under 2 seconds for real-time scoring.

## Known Unknowns / Risks
- Incomplete or biased historical data may affect model accuracy.
- External economic shifts (e.g., interest rate changes, recessions) may impact model performance.
- Integration challenges with existing loan management systems.
- Risk of overfitting to past lending patterns that may embed bias.

## Lifecycle Mapping 
- Understand SME loan risk factors → Problem Framing & Scoping (Stage 01) → Scoping paragraph & stakeholder memo.  
- Collect and clean historical loan data → Data Collection & Cleaning (Stage 02) → Cleaned dataset in data.  
- Develop and validate predictive model → Modeling & Evaluation (Stage 03) → Trained model & performance report.  
- Deploy model for loan decision support → Deployment (Stage 04) → API/Dashboard tool for analysts.

## Repo Plan
/data - Raw and processed SME loan data  
/src - Python scripts for data preprocessing, feature engineering, and model training  
/notebooks - Jupyter notebooks for exploratory data analysis, modeling, and evaluation  
/docs - Project documentation, stakeholder memos, and presentation slides  
Cadence for updates - Weekly commits with milestone updates at each project stage

## Data Storage

This notebook demonstrates a reproducible data storage workflow using CSV and Parquet formats, managed via environment variables.

### Folder Structure

- `data/raw/` — Raw CSV files
- `data/processed/` — Optimized Parquet files

### Formats Used

- **CSV**: Human-readable and easy to share
- **Parquet**: Efficient for analytics, preserves data types

### Environment Variables

Paths are loaded from `.env`:


