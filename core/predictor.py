from sklearn.metrics import accuracy_score, r2_score

def evaluate_model(model, X_test, y_test, problem_type):

    predictions = model.predict(X_test)

    if problem_type == "Classification":
        score = accuracy_score(y_test, predictions)
    else:
        score = r2_score(y_test, predictions)

    return predictions, round(score, 4)