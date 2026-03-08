import pandas as pd


def clean_data(df):

    df = df.copy()
    cleaning_log = []

    total_cells = df.shape[0] * df.shape[1]
    missing_before = df.isnull().sum().sum()

    # --------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------

    dup_count = df.duplicated().sum()

    if dup_count > 0:
        df = df.drop_duplicates()
        cleaning_log.append(f"Removed {dup_count} duplicate rows")
    else:
        cleaning_log.append("No duplicate rows found")

    # --------------------------------------------------
    # Handle missing values
    # --------------------------------------------------

    for col in df.columns:

        missing = df[col].isnull().sum()

        if missing == 0:
            continue

        missing_ratio = missing / len(df)

        # Warning for high missing values
        if missing_ratio > 0.4:
            cleaning_log.append(
                f"⚠ Column '{col}' had high missing values ({round(missing_ratio*100,1)}%)"
            )

        # -------------------------
        # Numeric columns
        # -------------------------

        if pd.api.types.is_numeric_dtype(df[col]):

            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

            cleaning_log.append(
                f"{col}: Filled {missing} missing values using median"
            )

        # -------------------------
        # Categorical columns
        # -------------------------

        else:

            mode_val = df[col].mode()

            if len(mode_val) > 0:
                fill_val = mode_val[0]
            else:
                fill_val = "Unknown"

            df[col] = df[col].fillna(fill_val)

            cleaning_log.append(
                f"{col}: Filled {missing} missing values using mode"
            )

    # --------------------------------------------------
    # Reset index
    # --------------------------------------------------

    df = df.reset_index(drop=True)

    cleaning_log.append("Dataset index reset")

    # --------------------------------------------------
    # Data Quality Score
    # --------------------------------------------------

    missing_after = df.isnull().sum().sum()

    quality_score = round(
        (1 - (missing_after / total_cells)) * 100, 2
    )

    cleaning_log.append(f"Data Quality Score: {quality_score}%")

    return df, cleaning_log