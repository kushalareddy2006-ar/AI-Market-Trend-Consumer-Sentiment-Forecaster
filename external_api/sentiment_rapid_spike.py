import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from external_api import sentiment_rapid_spike
import notification.notification as notification


# -----------------------------
# API CONFIG
# -----------------------------
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
}

SEARCH_URL = "https://real-time-amazon-data.p.rapidapi.com/search"
REVIEW_URL = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"

COUNTRY = "US"


# -----------------------------
# CATEGORY KEYWORDS
# -----------------------------
CATEGORY_KEYWORDS = {
    "Electricals_Power_Backup": ["inverter", "ups", "power backup", "generator"],
    "Home_Appliances": ["air conditioner", "refrigerator", "washing machine"],
    "Kitchen_Appliances": ["mixer", "grinder", "microwave"],
    "Computers_Tablets": ["laptop", "tablet"],
    "Mobile_Accessories": ["charger", "earphones", "power bank"],
    "Wearables": ["smartwatch", "fitness band"],
    "TV_Audio_Entertainment": ["smart tv", "speaker"],
}


# -----------------------------
# SENTIMENT MODEL (same as news)
# -----------------------------
MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_sentiment(text):
    if pd.isna(text) or text.strip() == "":
        return "Neutral"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment_idx = torch.argmax(probs).item()

    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    return label_map[sentiment_idx]


# -----------------------------
# SEARCH PRODUCTS
# -----------------------------
def search_products(query):
    params = {"query": query, "page": 1, "country": COUNTRY}
    response = requests.get(SEARCH_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("data", {}).get("products", [])


def fetch_reviews(asin):
    params = {"asin": asin, "country": COUNTRY, "page": 1}
    response = requests.get(REVIEW_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("data", {}).get("reviews", [])


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def get_amazon_reviews():

    all_reviews = []

    for category, keywords in tqdm(CATEGORY_KEYWORDS.items(), desc="Categories"):
        for keyword in keywords:
            try:
                products = search_products(keyword)

                for product in products[:5]:
                    asin = product.get("asin")
                    if not asin:
                        continue

                    reviews = fetch_reviews(asin)

                    for r in reviews:
                        all_reviews.append({
                            "category": category,
                            "asin": asin,
                            "review_title": r.get("review_title"),
                            "review_text": r.get("review_text"),
                            "review_date": r.get("review_date"),
                            "collected_at": datetime.utcnow()
                        })

            except Exception as e:
                print(f"Error for keyword {keyword}: {e}")

    df = pd.DataFrame(all_reviews)
    df.drop_duplicates(subset=["asin", "review_text"], inplace=True)

    # -----------------------------
    # SENTIMENT
    # -----------------------------
    df["combined_text"] = (
        df["review_title"].fillna("") + ". " +
        df["review_text"].fillna("")
    )

    tqdm.pandas()
    df["sentiment_label"] = df["combined_text"].progress_apply(get_sentiment)
    df.drop(columns=["combined_text"], inplace=True)

    # -----------------------------
    # SENTIMENT SPIKE
    # -----------------------------
    alert_df = sentiment_rapid_spike.new_sentiment_spike(df)

    if alert_df.empty:
        notification.send_mail(
            "Amazon Reviews Alert",
            "Amazon data extracted successfully. No major sentiment spikes or trend shifts detected."
        )
    else:
        notification.send_mail(
            "Amazon Reviews Alert",
            "Amazon sentiment spike or trend shift detected. Please find attached report.",
            alert_df
        )

    # -----------------------------
    # SAVE FINAL DATA
    # -----------------------------
    df.to_csv("final data/amazon_reviews_with_sentiment.csv", index=False)

    print("✅ RapidAPI sentiment pipeline completed")


if __name__ == "__main__":
    get_amazon_reviews()
