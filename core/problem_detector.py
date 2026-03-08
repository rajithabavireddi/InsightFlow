def detect_problem_type(target_series):

    if target_series.nunique() <= 15:
        return "Classification"
    else:
        return "Regression"