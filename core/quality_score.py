def calculate_quality_score(df):

    total_cells = df.shape[0] * df.shape[1]
    missing_ratio = df.isnull().sum().sum() / total_cells
    duplicate_ratio = df.duplicated().sum() / df.shape[0]

    score = 100 - (missing_ratio * 50) - (duplicate_ratio * 30)
    return round(max(score, 0), 2)


def interpret_score(score):

    if score >= 95:
        return "Excellent – Production Ready"
    elif score >= 85:
        return "Good – Minor preprocessing required"
    elif score >= 70:
        return "Moderate – Significant cleaning required"
    elif score >= 50:
        return "Poor – High risk dataset"
    else:
        return "Critical – Not suitable for ML"