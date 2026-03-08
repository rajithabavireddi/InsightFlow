def generate_recommendation(best_model, best_score, risk, readiness):

    return f"""
Model Selected: {best_model}

Cross Validation Score: {round(best_score, 4)}

Risk Level: {risk}

Deployment Readiness Score: {readiness}%

Recommendation:
• Proceed to pilot deployment if business validation confirms.
• Monitor model drift and retrain periodically.
• Ensure regulatory and compliance review before production rollout.
"""