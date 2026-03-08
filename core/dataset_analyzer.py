def analyze_dataset(df):

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Numeric Features": len(df.select_dtypes(include=["int64", "float64"]).columns),
        "Categorical Features": len(df.select_dtypes(include=["object"]).columns)
    }