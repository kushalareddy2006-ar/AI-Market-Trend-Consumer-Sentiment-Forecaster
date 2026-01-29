import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# config 
INPUT_FILE="sentiment_categorized_products.csv"
OUTPUT_FILE="category_wise_lda_output.csv"
NUM_TOPICS_PER_CATEGORY=5
MIN_DOCS_PER_CATEGORY=20
MAX_DF=0.8   #remove words that appear in more than 80% of document
MIN_DF=5    # keeps words that appear in at least 5 documents
TOP_WORDS=10 #Number of top keywords shown per topic
RANDOM_STATE=42 #randomness in LDA result 


# Custom stopwords 

CUSTOM_STOPWORDS = set([
    "good", "bad", "excellent", "poor", "amazing", "nice", "worst", "best",
    "love", "hate", "perfect", "terrible", "awesome", "waste",
    "money", "worth", "value", "price",
    "product", "products", "quality", "buy", "purchase",
    "using", "use", "used", "really", "very", "highly",
    "recommend", "recommended", "work", "works", "working", "flipkart"
])


# text cleaning 

def clean_for_lda(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]"," ",text)  #remove number, punctuation, symbols
    tokens = text.split()
    tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS and len(t)>2]
    return " ".join(tokens)

# load data 

df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=["cleaned_text", "category"])
df = df[df["cleaned_text"].str.strip()!=""]



# apply additional cleaning for LDA
df["lda_text"]= df["cleaned_text"].apply(clean_for_lda)

print("Total records:", len(df))


# helper: Extract Topic Words 
def get_topic_words(model, feature_names, top_n):
    topic_map = {}
    for idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-top_n:][::-1]
        words = [feature_names[i] for i in top_indices]
        topic_map[idx] = ", ".join(words)
    return topic_map



# store result 
final_results=[]


# category-wise LDA 

for category, df_cat in df.groupby("category"):
    if len(df_cat)<MIN_DOCS_PER_CATEGORY:
        print(f"skipping {category} (only {len(df_cat)} reviews)")
        continue
    
    print(f"\n Running LDA for category: {category} ({len(df_cat)} reviews)")
    
    
    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=MAX_DF,
        min_df=MIN_DF,
        ngram_range=(1,2)   #capture phrases like "battery life"
    )
    
    doc_term_matrix = vectorizer.fit_transform(df_cat["lda_text"])
    
    if doc_term_matrix.shape[1]<NUM_TOPICS_PER_CATEGORY:
        print(f"skipping {category} (not enough unique terms)")
        continue
    
    lda = LatentDirichletAllocation(
        n_components=NUM_TOPICS_PER_CATEGORY,
        random_state=RANDOM_STATE,
        learning_method="batch",
        max_iter=20
    )
    
    lda.fit(doc_term_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    topic_word_map = get_topic_words(lda, feature_names, TOP_WORDS)
    
    print(f"\n Topics discovered for category: {category}")
    for topic_id, keywords in topic_word_map.items():
        print(f"Topic {topic_id}: {keywords}")
        
        
    
    topic_dist = lda.transform(doc_term_matrix)
    
    df_cat = df_cat.copy()
    df_cat['lda_topic']= topic_dist.argmax(axis=1)
    df_cat["lda_topic_confidance"]= topic_dist.max(axis=1)
    df_cat["lda_topic_keywords"]= df_cat["lda_topic"].map(topic_word_map)
    
    
    final_results.append(df_cat)
    
    
# combine and save

if final_results:
    final_df = pd.concat(final_results, ignore_index=True)  
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n Saved result to: {OUTPUT_FILE}") 
else:
    print("No category had enough data for LDA")
    