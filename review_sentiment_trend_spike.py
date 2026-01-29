import pandas as pd
from notification import notification
csv_path = "final data/category_wise_lda_output_with_topic_labels.csv"

DATE_COL = "review_date"
SENTIMENT_COL = "sentiment_label"
CATEGORY_COL= "category"


WEEK_WINDOW = 2       # rolling over 2 weeks
SPIKE_THRESHOLD =  0.3        # 0 to 1      week 1: review = 0.7     week 2: review = 0.5   0.5-0.7  = -0.2 (considered)
TREND_SHIFT_THRESHOLD = 0.3


LABEL_MAP={
    "Positive":1,
    "Neutral":0,
    "Negative":-1
}


# load and clean data 

df = pd.read_csv(csv_path)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors = "coerce")
df = df.dropna(subset=[DATE_COL, SENTIMENT_COL, CATEGORY_COL])
df = df.sort_values(DATE_COL)

# map sentiment to numeric

df["sentiment_score"] = df[SENTIMENT_COL].map(LABEL_MAP)

# weekly aggregation (category wise)

weekly_cat = (
    df.groupby([CATEGORY_COL, df[DATE_COL].dt.to_period("W")])["sentiment_score"]
    .mean()
    .reset_index(name="weekly_sentiment")
)


weekly_cat[DATE_COL] = weekly_cat[DATE_COL].dt.start_time
weekly_cat = weekly_cat.sort_values([CATEGORY_COL, DATE_COL])


# Rolling window calculation

weekly_cat["rolling_avg"]=(
    weekly_cat
    .groupby(CATEGORY_COL)["weekly_sentiment"]
    .rolling(WEEK_WINDOW, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# shifting by 1 week 
weekly_cat["prev_rolling_avg"] = weekly_cat.groupby(CATEGORY_COL)['rolling_avg'].shift(1)
weekly_cat["delta"] = weekly_cat["rolling_avg"]-weekly_cat["prev_rolling_avg"]
weekly_cat["prev_delta"]= weekly_cat.groupby(CATEGORY_COL)["delta"].shift(1)


# calculate shift 

latest_week = weekly_cat[DATE_COL].max()
last_week = latest_week-pd.Timedelta(weeks=1)

weekly_cat_recent = weekly_cat[
    weekly_cat[DATE_COL]>=last_week
]

# alert detection 
alerts =[]

for _, row in weekly_cat_recent.iterrows():
    if pd.isna(row["delta"]):
        continue
    
    # negative spike 
    if row["delta"]<= -SPIKE_THRESHOLD:
        alerts.append({
            "date": row[DATE_COL],
            "category":row[CATEGORY_COL],
            "alert_type": "NEGATIVE SPIKE",
            "change": round(row["delta"], 3)
        })
        
    # positive spike 
    if row["delta"]>= SPIKE_THRESHOLD:
        alerts.append({
            "date": row[DATE_COL],
            "category":row[CATEGORY_COL],
            "alert_type": "POSITIVE SPIKE",
            "change": round(row["delta"], 3)
        })
        
    # trend shift (direction flip)
    if(
        pd.notna(row['prev_delta']) and 
        (row["delta"]*row['prev_delta']<0) and      # -2 x -2  = 4 (positive)   2 x 2 = 4 (postive)     2 x -2 = -4
        abs(row["delta"])>=TREND_SHIFT_THRESHOLD
    ):
        alerts.append({
            "date": row[DATE_COL],
            "category":row[CATEGORY_COL],
            "alert_type": "TREND SHIFT",
            "change": round(row["delta"], 3)
        })
        
        
# output 

alert_df = pd.DataFrame(alerts)

if alert_df.empty:
    print("No trend shift and spike")
    
else:
    print("trend shift")
    print(alert_df.sort_values(['date', 'category']).reset_index(drop=True))
    


notification.send_mail(subject="Sentiment Spike and Trend Shift Notification", text="Please find the attached file of sentiment analysis and trend shify of Reviews data", df=alert_df)
# notification.send_mail(subject="Hello", text="Hai", df=alert_df)