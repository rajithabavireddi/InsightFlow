def select_best_model(cv_scores):

    best_model = max(cv_scores, key=cv_scores.get)
    best_score = cv_scores[best_model]

    return best_model, best_score