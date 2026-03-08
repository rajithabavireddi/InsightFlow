import pandas as pd

def make_arrow_compatible(df):

    df = df.copy()

    for col in df.columns:

        # convert mixed object columns to string
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

        # convert category to string
        if str(df[col].dtype) == "category":
            df[col] = df[col].astype(str)

    return df