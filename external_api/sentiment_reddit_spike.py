import pandas as pd

def reddit_sentiment_spike(df):
    # -----------------------------
    # CONFIG
    # -----------------------------
    DATE_COL = "created_date"
    SENTIMENT_COL = "sentiment_label"
    CATEGORY_COL = "category_label"

    WEEK_WINDOW = 2
    SPIKE_THRESHOLD = 0.3
    TREND_SHIFT_THRESHOLD = 0.3

    LABEL_MAP = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    # -----------------------------
    # CLEAN & PREP DATA
    # -----------------------------

    # ✅ FIX 1: Parse correct datetime format
    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"
    )

    # Normalize sentiment labels (VERY IMPORTANT)
    df[SENTIMENT_COL] = df[SENTIMENT_COL].str.strip().str.lower()

    df = df.dropna(subset=[DATE_COL, SENTIMENT_COL, CATEGORY_COL])
    df = df.sort_values(DATE_COL)

    # Map sentiment to numeric
    df["sentiment_score"] = df[SENTIMENT_COL].map(LABEL_MAP)

    # -----------------------------
    # WEEKLY AGGREGATION
    # -----------------------------
    weekly_cat = (
        df.groupby([CATEGORY_COL, df[DATE_COL].dt.to_period("W")])["sentiment_score"]
        .mean()
        .reset_index(name="weekly_sentiment")
    )

    weekly_cat[DATE_COL] = weekly_cat[DATE_COL].dt.start_time

    # -----------------------------
    # ROLLING WINDOW CALCS
    # -----------------------------
    weekly_cat["rolling_avg"] = (
        weekly_cat
        .groupby(CATEGORY_COL)["weekly_sentiment"]
        .rolling(WEEK_WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    weekly_cat["prev_rolling_avg"] = (
        weekly_cat.groupby(CATEGORY_COL)["rolling_avg"].shift(1)
    )

    weekly_cat["delta"] = weekly_cat["rolling_avg"] - weekly_cat["prev_rolling_avg"]
    weekly_cat["prev_delta"] = (
        weekly_cat.groupby(CATEGORY_COL)["delta"].shift(1)
    )

    # -----------------------------
    # DETECT ALERTS
    # -----------------------------
    alerts = []

    for _, row in weekly_cat.iterrows():
        if pd.isna(row["delta"]):
            continue

        if row["delta"] <= -SPIKE_THRESHOLD:
            alerts.append({
                "date": row[DATE_COL],
                "category": row[CATEGORY_COL],
                "type": "NEGATIVE SPIKE",
                "change": round(row["delta"], 3)
            })

        if row["delta"] >= SPIKE_THRESHOLD:
            alerts.append({
                "date": row[DATE_COL],
                "category": row[CATEGORY_COL],
                "type": "POSITIVE SPIKE",
                "change": round(row["delta"], 3)
            })

        if (
            pd.notna(row["prev_delta"])
            and row["delta"] * row["prev_delta"] < 0
            and abs(row["delta"]) >= TREND_SHIFT_THRESHOLD
        ):
            alerts.append({
                "date": row[DATE_COL],
                "category": row[CATEGORY_COL],
                "type": "TREND SHIFT",
                "change": round(row["delta"], 3)
            })

    alert_df = pd.DataFrame(alerts)

    # -----------------------------
    # ✅ FIX 2: FILTER ONLY LAST WEEK
    # -----------------------------
    if not alert_df.empty:
        latest_date = alert_df["date"].max()
        last_week_start = latest_date - pd.Timedelta(days=7)

        alert_df = alert_df[
            alert_df["date"] >= last_week_start
        ]

    # -----------------------------
    # OUTPUT
    # -----------------------------
    if alert_df.empty:
        print("✅ No Reddit sentiment spikes in the last week.")
    else:
        print("🚨 LAST WEEK REDDIT SENTIMENT ALERTS")
        print(
            alert_df
            .sort_values(["date", "category"])
            .reset_index(drop=True)
        )

    return alert_df


# if __name__ == "__main__":
#     CSV_PATH = "../final data/reddit_category_trend_data.xlsx"
#     df = pd.read_excel(CSV_PATH)
#     reddit_sentiment_spike(df)
