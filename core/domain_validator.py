def validate_domain(df, selected_domain):

    if selected_domain == "General":
        return True

    keywords = {
        "Healthcare": ["age", "blood", "patient", "diagnosis", "cholesterol"],
        "Finance": ["transaction", "amount", "account", "balance", "fraud"],
        "Real Estate": ["price", "area", "bedroom", "bathroom", "location"]
    }

    domain_keywords = keywords.get(selected_domain, [])
    columns = [col.lower() for col in df.columns]

    match_count = sum(
        any(keyword in col for keyword in domain_keywords)
        for col in columns
    )

    return match_count >= 2