




import pandas as pd
from transformers import pipeline
import torch
import re


df = pd.read_csv("reduced_combined_cleaned_data.csv")

def clean_product_name(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

df['cleaned_product'] = df['product'].apply(clean_product_name)


# split fipkart and non flipkart data 
df_flipkart = df[df['source'].str.lower() == 'flipkart']
df_non_flipkart = df[df['source'].str.lower() != 'flipkart']


# Load zero-shot-classification pipeline

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli",
                      device=device
                      )


labels = [
    "Electricals_Power_Backup",
    "Home_Appliances",
    "Kitchen_Appliances",
    "Furniture",
    "Home_Storage_Organization",
    "Computers_Tablets",
    "Mobile_Accessories",
    "Wearables",
    "TV_Audio_Entertainment",
    "Networking_Devices",
    "Toys_Kids",
    "Gardening_Outdoor",
    "Kitchen_Dining",
    "Mens_Clothing",
    "Footwear",
    "Beauty_Personal_Care",
    "Security_Surveillance",
    "Office_Printer_Supplies",
    "Software",
    "Fashion_Accessories",
]

# deduplicate flipkart products 

unique_products =(
    df_flipkart[["product", "cleaned_product"]]
    .dropna()
    .drop_duplicates(subset=["cleaned_product"])
    .to_dict("records")
)


print(f"Total unique flipkart products to classify: {len(unique_products)}")

# print(unique_products)

# rule-based keyword override

def keyword_override(product_name):
    if "juicer" in product_name or "mixer" in product_name or "grinder" in product_name:
        return "Kitchen_Appliances"
    if "charger" in product_name or "cable" in product_name or "cover" in product_name:
        return "Mobile_Accessories"
    if "toy" in product_name or "kids" in product_name or "puzzle" in product_name:
        return "Toys_Kids"
    return None



# batch classify

batch_size = 16
pred_rows = []


#   1 batch 0 to 16
#  2 batch 17 to 32
for i in range(0, len(unique_products), batch_size):
    batch = unique_products[i : i + batch_size]
    texts = [item["cleaned_product"] for item in batch]
    results = classifier(texts, labels)
    
    if isinstance(results, dict):
        results = [results]
        
    for item, res in zip(batch, results):
        override_category = keyword_override(item["cleaned_product"])
        
        final_category = (
            override_category if override_category else res["labels"][0]
        
        )
        final_confidence =(
            1.0 if override_category else round(float(res["scores"][0]), 3)
        )
        
        pred_rows.append(
            {
                "product": item["product"],
                "category": final_category,   # ← directly category line has been changed
                "confidence": final_confidence,
            }
        )
        
        
pred_df = pd.DataFrame(pred_rows)        

# merge prediction back 
df_flipkart = df_flipkart.drop(columns=["category"], errors="ignore")  # added this line 

df_flipkart = df_flipkart.merge(
    pred_df, on="product", how="left"
)

# non flipkart 
# df_non_flipkart["pred_category"] = None      remove this line 
df_non_flipkart["confidence"] = None

# combine and save
df_final = pd.concat([df_flipkart, df_non_flipkart], ignore_index=True)

df_final.to_csv("categorized_products1.csv", index=False)

print("Categorization completed and saved to categorized_products.csv")