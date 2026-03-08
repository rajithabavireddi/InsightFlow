def build_report(dataset_info, quality_score, best_model, model_score, risk, readiness):

    report = f"""
===== INSIGHTFLOW ANALYTICS REPORT =====

Dataset Summary:
Rows: {dataset_info['Rows']}
Columns: {dataset_info['Columns']}

Data Quality Score: {quality_score}

Selected Model: {best_model}
Model Performance: {model_score}

Risk Level: {risk}
Deployment Readiness: {readiness}%

========================================
"""

    return report