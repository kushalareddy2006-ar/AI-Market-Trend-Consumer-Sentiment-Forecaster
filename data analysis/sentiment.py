import pandas as pd
import numpy as np


# load data 
df = pd.read_csv("categorized_products1.csv")


# rating based sentiment for flipkart 

def rating_sentiment(row):
    if str(row['source']).lower()=='flipkart':
        try:
            rating = int(row['rating'])
        except:
            return "unknown"
        if rating in [1,2]:
            return "Negative"
        elif rating == 3:
            return "Neutral"
        elif rating in [4,5]:
            return "Positive"
        else:
            return "unknown"
    else:
        return row.get("sentiment_label", None)  #keep previous sentiment for non-flipkart
    
df['sentiment_label'] = df.apply(rating_sentiment, axis=1)  

# add random review_date (2021-01-01 to 2025-01-01)

start_date = pd.to_datetime('2021-01-01')
end_date = pd.to_datetime('2025-01-01')

# in mask keep all dates that are NaN or empty
mask = df['review_date'].isna() | (df['review_date'] == '')


random_dates = pd.to_datetime(
    np.random.randint(
        start_date.value // 10**9,   #convert date into a integer timestamp
        end_date.value // 10**9,
        size=mask.sum()
    ),
    unit='s'
)

df.loc[mask, 'review_date'] = random_dates.strftime('%m/%d/%Y')

# save the updated dataframe
df.to_csv("sentiment_categorized_products.csv", index=False)

print("Sentiment labeling and review date imputation completed and saved to 'sentiment_categorized_products.csv'.")