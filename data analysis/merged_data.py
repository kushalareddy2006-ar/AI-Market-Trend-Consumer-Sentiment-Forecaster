import pandas as pd
import re
import nltk
from nltk.corpus import stopwords


# # download NLTK stopwords  
nltk.download("stopwords")
stop_words= set(stopwords.words("english"))


# cleaning function  
def clean_lowercase(text):
    if isinstance(text, str):
        return text.lower()
    return text

def clean_punctuation(text):
    if isinstance(text, str):
        return re.sub(r"[^\w\s]","",text)
    return text

def clean_stopwords(text):
    if isinstance(text, str):
        return " ".join([w for w in text.split() if w not in stop_words])
    return text

def clean_whitespace(text):
    if isinstance(text, str):
        return re.sub(r"\s+"," ",text).strip()
    return text

def clean_text(text):
    text = clean_lowercase(text)
    text = clean_punctuation(text)
    text = clean_stopwords(text)
    text = clean_whitespace(text)
    return text


# load files  
df_flip = pd.read_csv("data/flipkart_product.csv", encoding="latin1", engine="python")

df_amazon = pd.read_excel("data/Amazon DataSheet - Pradeep.xlsx")

def clean_text_flip(t):
    t=str(t)
    
    t=re.sub(r"[^\x00-\x7F]"," ",t)
    t=re.sub(r"^a-zA-Z0-9"," ",t)
    
    t=re.sub(r"\s+"," ",t)
    return t.strip()

df_flip = df_flip.applymap(lambda x:clean_text_flip(x) if isinstance(x,str) else x)


# standardize flipkart columns

df_flip = df_flip.rename(columns={
    "ProductName" : "product",
    "Review": "review_title",
    "Summary": "review_text",
    "Rate":"rating"
})


df_flip["source"]="flipkart"
df_flip["review_date"]=""
df_flip["sentiment_lable"]=""
df_flip["category"]=""

# standardize amazon columns

df_amazon = df_amazon.rename(columns={
    "Product Name" : "product",
    "User Review": "review_text",
    "Star Rating":"rating",
    "Date of Review":"review_date",
    "Category":"category",
    "Sentiment":"sentiment_lable"
})

df_amazon["source"]="amazon"
df_amazon["review_title"]=""


# keep only required columns
required = [
    "source",
    "product",
    "review_text",
    "review_title",
    "rating",
    "category",
    "review_date",
    "sentiment_lable"
]

df_flip = df_flip[required]
df_amazon = df_amazon[required]

# combine both
df = pd.concat([df_flip,df_amazon], ignore_index=True)


# clean review_text
df["cleaned_text"]=df["review_text"].astype(str).apply(clean_text)

# clean prduct name 
df["product"]=df["product"].astype(str).apply(clean_text)


# add sentiment_score
df["sentiment_score"]=""

# save ouput
df.to_csv("combined_cleaned_data.csv", index=False)

print("Final dataset saved as combined_cleaned_data.csv")