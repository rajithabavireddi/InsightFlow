def calculate_deployment_score(quality_score, model_score):

    final_score = (quality_score * 0.3) + (model_score * 100 * 0.7)
    return round(final_score, 2)