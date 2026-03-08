def calculate_risk(model_score):

    if model_score >= 0.9:
        return "Low Risk"
    elif model_score >= 0.75:
        return "Medium Risk"
    else:
        return "High Risk"