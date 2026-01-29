import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
# from dotenv import load_dotenv

# load_dotenv()

# -----------------------------
# API CONFIG
# -----------------------------
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_KEY = "23f7893084mshcb0c089220fe115p1d0ff1jsn00fe50a253e9"
HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
}

SEARCH_URL = "https://real-time-amazon-data.p.rapidapi.com/search"
REVIEW_URL = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"

COUNTRY = "US"
SEARCH_PAGE = 1
REVIEW_PAGE = 1

# -----------------------------
# CATEGORY KEYWORDS (UNCHANGED)
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
# SEARCH AMAZON PRODUCTS
# -----------------------------
def search_products(query):
    params = {
        "query": query,
        "page": SEARCH_PAGE,
        "country": COUNTRY,
        "sort_by": "RELEVANCE",
        "product_condition": "ALL",
        "is_prime": "false",
        "deals_and_discounts": "NONE"
    }

    response = requests.get(SEARCH_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("data", {}).get("products", [])

# -----------------------------
# FETCH REVIEWS BY ASIN
# -----------------------------
def fetch_reviews(asin):
    params = {
        "asin": asin,
        "country": COUNTRY,
        "page": REVIEW_PAGE,
        "sort_by": "TOP_REVIEWS",
        "star_rating": "ALL",
        "verified_purchases_only": "false",
        "images_or_videos_only": "false",
        "current_format_only": "false"
    }

    response = requests.get(REVIEW_URL, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json().get("data", {}).get("reviews", [])

# -----------------------------
# MAIN PIPELINE
# -----------------------------
all_reviews = []

for category, keywords in tqdm(CATEGORY_KEYWORDS.items(), desc="Categories"):
    for keyword in keywords:
        try:
            products = search_products(keyword)

            for product in products[:5]:  # limit to avoid rate-limit
                asin = product.get("asin")
                if not asin:
                    continue

                reviews = fetch_reviews(asin)

                for r in reviews:
                    all_reviews.append({
                        "category": category,
                        "keyword_used": keyword,
                        "asin": asin,
                        "product_title": product.get("title"),
                        "brand": product.get("brand"),
                        "price": product.get("price"),
                        "rating": r.get("rating"),
                        "review_title": r.get("review_title"),
                        "review_text": r.get("review_text"),
                        "review_date": r.get("review_date"),
                        "reviewer": r.get("reviewer_name"),
                        "verified_purchase": r.get("verified_purchase"),
                        "collected_at": datetime.utcnow()
                    })

        except Exception as e:
            print(f"Error for keyword '{keyword}': {e}")

# -----------------------------
# SAVE DATA
# -----------------------------
df = pd.DataFrame(all_reviews)

df.drop_duplicates(
    subset=["asin", "review_text"],
    inplace=True
)

df.to_csv("amazon_reviews_categorized.csv", index=False)

print(f"Saved {len(df)} reviews to amazon_reviews_categorized.csv")

