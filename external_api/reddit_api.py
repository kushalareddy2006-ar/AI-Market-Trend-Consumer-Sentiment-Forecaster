# import requests
# import pandas as pd
# from datetime import datetime
# import time

# # -----------------------------
# # Labels (Categories)
# # -----------------------------
# labels = [
#     "Electricals_Power_Backup",
#     "Home_Appliances",
#     "Kitchen_Appliances",
#     "Furniture",
#     "Home_Storage_Organization",
#     "Computers_Tablets",
#     "Mobile_Accessories",
#     "Wearables",
#     "TV_Audio_Entertainment",
#     "Networking_Devices",
#     "Toys_Kids",
#     "Gardening_Outdoor",
#     "Kitchen_Dining",
#     "Mens_Clothing",
#     "Footwear",
#     "Beauty_Personal_Care",
#     "Security_Surveillance",
#     "Office_Printer_Supplies",
#     "Software",
#     "Fashion_Accessories",
# ]

# # -----------------------------
# # Reddit API
# # -----------------------------
# url = "https://www.reddit.com/search.json"
# headers = {
#     "User-Agent": "Mozilla/5.0 (ConsumerTrendAnalysisBot)"
# }


# all_rows = [ ]

# # -----------------------------
# # Fetch Reddit data
# # -----------------------------
# for label in labels:
#     print(f"Fetching data for: {label}")

#     params = {
#         "q": label.replace("_", " "),
#         "limit": 100
#     }

#     response = requests.get(url, headers=headers, params=params)

#     if response.status_code != 200:
#         print(f"Failed for {label}")
#         continue

#     data = response.json()

#     for post in data.get("data", {}).get("children", []):
#         post_data = post.get("data", {})

#         all_rows.append({
#             "source": "Reddit",
#             "category_label": label,
#             "query": label.replace("_", " "),
#             "title": post_data.get("title", ""),
#             "selftext": post_data.get("selftext", ""),
#             "subreddit": post_data.get("subreddit", ""),
#             "score": post_data.get("score", 0),
#             "num_comments": post_data.get("num_comments", 0),
#             "created_date": datetime.utcfromtimestamp(
#                 post_data.get("created_utc", 0)
#             )
#         })

#     # polite delay
#     time.sleep(2)

# # -----------------------------
# # Create DataFrame
# # -----------------------------
# df = pd.DataFrame(all_rows)

# # Remove empty or NaN text safely
# df = df[df["selftext"].str.strip() != ""].reset_index(drop=True)

# print("Total records collected:", len(df))
# print(df.head())

# # -----------------------------
# # Save Output (CSV)
# # -----------------------------
# output_file = "reddit_mixed_consumer_data.xlsx"
# df.to_excel(output_file, index=False)

# print(f"Mixed dataset saved to: {output_file}")

import requests
import pandas as pd
from datetime import datetime
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from notification import notification
import os
from external_api import sentiment_reddit_spike
tqdm.pandas()

# -----------------------------
# Reddit Search Map
# -----------------------------
reddit_search_map = {
    "Electricals_Power_Backup": "inverter OR UPS OR power backup",
    "Home_Appliances": "home appliances OR washing machine OR refrigerator",
    "Kitchen_Appliances": "kitchen appliances OR air fryer OR mixer grinder",
    "Furniture": "furniture OR sofa OR bed",
    "Home_Storage_Organization": "storage organizer OR wardrobe OR shelf",
    "Computers_Tablets": "laptop OR desktop OR tablet",
    "Mobile_Accessories": "mobile accessories OR phone case OR charger",
    "Wearables": "smartwatch OR fitness band OR wearable",
    "TV_Audio_Entertainment": "television OR smart TV OR speakers",
    "Networking_Devices": "router OR modem OR WiFi",
    "Toys_Kids": "kids toys OR baby toys OR educational toys",
    "Gardening_Outdoor": "gardening OR outdoor tools OR plants",
    "Kitchen_Dining": "kitchen dining OR cookware OR dinner set",
    "Mens_Clothing": "mens clothing OR men fashion OR shirts",
    "Footwear": "footwear OR shoes OR sneakers",
    "Beauty_Personal_Care": "beauty OR skincare OR personal care",
    "Security_Surveillance": "CCTV OR security camera OR surveillance",
    "Office_Printer_Supplies": "printer OR ink cartridge OR office supplies",
    "Software": "software OR apps OR technology",
    "Fashion_Accessories": "fashion accessories OR bags OR watches"
}

def reddit_api():
    try:
        # -----------------------------
        # Reddit API Config
        # -----------------------------
        headers = {
            "User-Agent": "Mozilla/5.0 (TrendAnalysisBot)"
        }

        BASE_URL = "https://www.reddit.com/search.json"
        all_rows = []

        # -----------------------------
        # SCRAPE REDDIT DATA
        # -----------------------------
        for label, query in reddit_search_map.items():
            print(f"🔍 Searching Reddit for: {label}")

            params = {
                "q": query,
                "limit": 100
            }

            response = requests.get(BASE_URL, headers=headers, params=params)
            if response.status_code != 200:
                continue

            data = response.json()

            for post in data["data"]["children"]:
                post_data = post["data"]

                all_rows.append({
                    "source": "Reddit",
                    "category_label": label,
                    "search_query": query,
                    "title": post_data.get("title", ""),
                    "selftext": post_data.get("selftext", ""),
                    "subreddit": post_data.get("subreddit", ""),
                    "score": post_data.get("score", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "created_date": datetime.utcfromtimestamp(
                        post_data.get("created_utc", 0)
                    )
                })

            time.sleep(2)

        # -----------------------------
        # CREATE DATAFRAME
        # -----------------------------
        df = pd.DataFrame(all_rows)
        df = df[df["selftext"].str.strip() != ""].reset_index(drop=True)

        # -----------------------------
        # SENTIMENT ANALYSIS (RoBERTa)
        # -----------------------------
        print("🤖 Loading sentiment model...")

        MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        LABELS = ["negative", "neutral", "positive"]
        NUMERIC_MAP = {"negative": -1, "neutral": 0, "positive": 1}

        # Combine text
        df["combined_text"] = (
            df["title"].fillna("") + " " + df["selftext"].fillna("")
        ).str.slice(0, 500)

        def get_sentiment(text):
            if not text.strip():
                return pd.Series(["neutral", 0.0, 0])

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

            label = LABELS[probs.argmax().item()]
            score = probs.max().item()
            numeric = NUMERIC_MAP[label]

            return pd.Series([label, score, numeric])

        print("📊 Running sentiment analysis...")
        
        
        df[["sentiment_label", "sentiment_score", "sentiment_numeric"]] = (
            df["combined_text"].progress_apply(get_sentiment)
        )

        # -----------------------------
        # SAVE FINAL EXCEL
        # -----------------------------
        
        df.to_excel("final data/reddit_category_trend_data.xlsx", index=False)
        print("✅ Saved reddit_category_trend_data.xlsx")
        
        # step 3
        result_df = sentiment_reddit_spike.reddit_sentiment_spike(df)
        if result_df.empty:
            notification.send_mail("News Data Alert", "News Data Extracted Successfully and No major weekly Reddit sentiment spikes or trend shifts detected." )
        else:
            notification.send_mail(subject="Extracted data from Reddit API", text=f"Successfully saved data from reddit API and please find the attached report of sentiment spike and trend shift of this week", df=result_df)
    except Exception as e:
        print("failed to save reddit data", e)
        notification.send_mail(subject="Reddit Data Extraction Failed", text=f"Failed to extract data from reddit api due to reason: {e}")
        
# if __name__ == "__main__":
#     reddit_api()  