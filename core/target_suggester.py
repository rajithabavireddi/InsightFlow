def suggest_target(df):

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) > 0:
        return numeric_cols[-1]

    return df.columns[-1]