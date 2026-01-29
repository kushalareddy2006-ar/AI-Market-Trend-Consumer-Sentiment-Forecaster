import pandas as pd
import re
import nltk
from nltk.corpus import stopwords


# download NLTK stopwords  
nltk.download("stopwords")
stop_words= set(stopwords.words("english"))

print(stop_words)


# load CSV file  
df = pd.read_csv("data/flipkart_product.csv", encoding="latin1", engine="python")


# drop exact duplicate rows
df = df.drop_duplicates()

# lowercase all text columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.lower()


# remove_punctuation
def remove_punctuation(text):
    if isinstance(text,str):
        return re.sub(r"[^\w\s]","",text)
    return text

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(remove_punctuation)
    
# remove stopwords 
def remove_stopwords(text):
    if isinstance(text, str):
        return " ".join(
            [word for word in text.split() if word not in stop_words]
        )
    return text

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].apply(remove_stopwords)
    
# remove extra spaces
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()

# save cleaned Output
df.to_csv("fipkart_product_cleaned.csv", index=False)
print("Data Cleaning Completed Successfully")