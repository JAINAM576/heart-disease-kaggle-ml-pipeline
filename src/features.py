def create_features(df):

    # Age × Max HR interaction
    df["Age_MaxHR"] = df["Age"] * df["Max HR"]

    # ST depression × Exercise angina
    df["ST_Angina"] = df["ST depression"] * df["Exercise angina"]

    # BP risk flag
    df["High_BP"] = (df["BP"] > 140).astype(int)

    # Vessel severity
    df["Severe_Vessel"] = (df["Number of vessels fluro"] >= 2).astype(int)

    return df
