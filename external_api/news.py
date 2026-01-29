import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import notification.notification as notification
from external_api import sentiment_news_spike
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


load_dotenv()
# -----------------------------
# API CONFIG
# -----------------------------
API_KEY = os.getenv("NEWS_API_KEY")

BASE_URL = "https://newsapi.org/v2/everything"


# i am collecting data form the last week so that i will have new data every week 
FROM_DATE = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")

LANGUAGE = "en"
PAGE_SIZE = 50

# -----------------------------
# CATEGORY KEYWORDS
# -----------------------------
CATEGORY_KEYWORDS = {
    "Electricals_Power_Backup": ["inverter", "ups", "power backup", "generator"],
    "Home_Appliances": ["air conditioner", "refrigerator", "washing machine", "air cooler"],
    "Kitchen_Appliances": ["mixer", "grinder", "microwave", "oven", "juicer"],
    "Furniture": ["sofa", "bed", "table", "chair"],
    "Home_Storage_Organization": ["storage box", "wardrobe", "organizer"],
    "Computers_Tablets": ["laptop", "tablet", "desktop"],
    "Mobile_Accessories": ["charger", "earphones", "power bank"],
    "Wearables": ["smartwatch", "fitness band"],
    "TV_Audio_Entertainment": ["smart tv", "speaker", "soundbar"],
    "Networking_Devices": ["router", "wifi modem"],
    "Toys_Kids": ["kids toys", "children games"],
    "Gardening_Outdoor": ["gardening", "lawn tools"],
    "Kitchen_Dining": ["cookware", "utensils"],
    "Mens_Clothing": ["mens clothing", "mens fashion"],
    "Footwear": ["shoes", "sneakers"],
    "Beauty_Personal_Care": ["skincare", "beauty products"],
    "Security_Surveillance": ["cctv", "security camera"],
    "Office_Printer_Supplies": ["printer", "scanner"],
    "Software": ["software", "saas"],
    "Fashion_Accessories": ["handbag", "watch", "wallet"]
}

# -----------------------------
# FETCH NEWS FUNCTION
# -----------------------------
def fetch_news(query, category):
    params = {
        "q": query,
        "from": FROM_DATE,
        "language": LANGUAGE,
        "sortBy": "popularity",
        "pageSize": PAGE_SIZE,
        "apiKey": API_KEY
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()

    articles = []
    for a in data.get("articles", []):
        articles.append({
            "source": a["source"]["name"],
            "author": a.get("author"),
            "title": a.get("title"),
            "description": a.get("description"),
            "content": a.get("content"),
            "url": a.get("url"),
            "image_url": a.get("urlToImage"),
            "published_at": a.get("publishedAt"),
            "category": category,
            "query_used": query,
            "collected_at": datetime.utcnow()
        })
    return articles

# -----------------------------
# SENTIMENT PREDICTION FUNCTION
# -----------------------------
def get_sentiment(text):
    
    MODEL_NAME = "ProsusAI/finbert"

    # -----------------------------
    # LOAD MODEL & TOKENIZER
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
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
# MAIN PIPELINE
# -----------------------------

def get_news_data():
    try:
        all_articles = []

        for category, keywords in tqdm(CATEGORY_KEYWORDS.items()):
            for keyword in keywords:
                try:
                    articles = fetch_news(keyword, category)
                    all_articles.extend(articles)
                except Exception as e:
                    print(f"Error fetching {keyword}: {e}")

        # -----------------------------
        # SAVE TO CSV
        # -----------------------------
        news_df = pd.DataFrame(all_articles)

        # Remove duplicates based on URL
        news_df.drop_duplicates(subset="url", inplace=True)

        # news_df.to_csv("news_data_categorized.csv", index=False)

        # print(f"Saved {len(news_df)} articles to news_data_categorized.csv")



        # -----------------------------
        # CONFIG
        # -----------------------------
        OUTPUT_FILE = "final data/news_data_with_sentiment.csv"



        # -----------------------------
        # COMBINE TEXT FIELDS
        # -----------------------------
        news_df["combined_text"] = (
            news_df["title"].fillna("") + ". " +
            news_df["description"].fillna("") + ". " +
            news_df["content"].fillna("")
        )

        # -----------------------------
        # APPLY SENTIMENT MODEL
        # -----------------------------
        tqdm.pandas()
        news_df["sentiment_label"] = news_df["combined_text"].progress_apply(get_sentiment)

        # -----------------------------
        # SAVE OUTPUT
        # -----------------------------
        news_df.drop(columns=["combined_text"], inplace=True)
        
        if os.path.exists(OUTPUT_FILE):
            # Append without header
            news_df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)
        else:
            # Create file with header
            news_df.to_csv(OUTPUT_FILE, index=False)
        
        result_df = sentiment_news_spike.new_sentiment_spike(news_df)
        if result_df.empty:
            notification.send_mail("News Data Alert", "News Data Extracted Successfully and No major weekly Reddit sentiment spikes or trend shifts detected." )
        else:
            notification.send_mail("News Data Alert", "News Data Extracted Successfully and please find the attached report of sentiment spike and trend shift of this week", result_df )
            print("✅ Sentiment analysis completed and saved successfully.")

        
        # news_df.to_csv("news_data_categorized.csv", index=False)

        # print(f"Saved {len(news_df)} articles to news_data_categorized.csv")
    except Exception as e:
        notification.send_mail("News Data Alert", f"Failed to Extract News Data and the reason is: {e}")
        
        
# if __name__ == "__main__":
#     get_news_data()      