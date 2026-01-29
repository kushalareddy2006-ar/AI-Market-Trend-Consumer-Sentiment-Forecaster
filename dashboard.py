import streamlit as st
import pandas as pd
import plotly.express as px

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai
from google.genai import types
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ----------------scheduler--------------
import schedule
import threading
import time
from external_api import reddit_api, news


def run_scheduler():
    
    # call function every 10 seconds 
    # schedule.every(10).seconds.do(notification.testing_function)
    
    # schedule.every(10).minutes.do(notification.testings_function)
    
    # schedule.every().day.at("21:32").do(news.get_news_data)
    schedule.every().sunday.at("02:00").do(news.get_news_data)
    schedule.every().sunday.at("03:00").do(reddit_api.reddit_api)

    
    # schedule.every(10).monday.at("02:00").do(notification.testing_function)
    
    while True:
        schedule.run_pending()
        time.sleep(5)
    
threading.Thread(target=run_scheduler, daemon=True).start()


# ----------------scheduler--------------
if __name__ == "__main__":
    # page config

    st.set_page_config(
        page_title = "AI Market Trend & Consumer Sentiment Forecaster",
        layout="wide"
    )

    st.title("AI-Powered Market Trend & Consumer Sentiment Dashboard")
    st.markdown("Consumer sentiment, topic trend, and social insights from reviews, news, and Reddit data")


    # load data 

    @st.cache_data
    def load_data():
        reviews = pd.read_csv("final data/category_wise_lda_output_with_topic_labels.csv")
        
        reddit = pd.read_excel("final data/reddit_category_trend_data.xlsx")
        
        news = pd.read_csv("final data/news_data_with_sentiment.csv")
        
        print(reddit.columns)
        if "review_date" in reviews.columns:
            reviews["review_date"]= pd.to_datetime(
                reviews["review_date"], errors="coerce"   #convert invalid date to NaT 
            )
            
        
        if "published_at" in news.columns:
            news["published_at"]= pd.to_datetime(
                news["published_at"], errors="coerce"   #convert invalid date to NaT 
            )
        
        if "created_date" in reddit.columns:
            reddit["created_date"]= pd.to_datetime(
                reddit["created_date"], errors="coerce"   #convert invalid date to NaT 
            )    
            
        return reviews, reddit, news   


    reviews_df, reddit_df, news_df  = load_data()

    # load vector database 
    @st.cache_resource
    def load_vector_db():
        # load embedding model 

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vector_db = FAISS.load_local(
            "consumer_sentiment_faiss",
            embeddings,
            allow_dangerous_deserialization = True
        )
        
        return vector_db

    vector_db = load_vector_db()


    # load gemini 
    @st.cache_resource
    def load_gemini_client():
        return genai.Client(api_key=os.getenv("Gemini_Api_key"))

    client = load_gemini_client()

    main_col, right_sidebar = st.columns([3,1])

    with main_col:
        # sidebar filters

        st.sidebar.header("Filters")

        source_filter = st.sidebar.multiselect(
            "Select Source",
            options=reviews_df["source"].unique(),
            default=reviews_df["source"].unique()
        )


        category_filter = st.sidebar.multiselect(
            "Select Category",
            options=reviews_df["category"].unique(),
            default=reviews_df["category"].unique()
        )

        filtered_reviews = reviews_df[
            (reviews_df["source"].isin(source_filter))&
            (reviews_df["category"].isin(category_filter))
        ]



        # KPI Metrics

        st.subheader("Key Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Reviews", len(filtered_reviews))
        col2.metric("Positive %", round((filtered_reviews["sentiment_label"]=="Positive").mean()*100,1))
        col3.metric("Negative %", round((filtered_reviews["sentiment_label"]=="Negative").mean()*100,1))
        col4.metric("Neutral %", round((filtered_reviews["sentiment_label"]=="Neutral").mean()*100,1))


        # sentiment distribution 

        col1, col2 = st.columns(2)

        with col1:
            sentiment_dist = filtered_reviews["sentiment_label"].value_counts().reset_index()
            sentiment_dist.columns=["Sentiment", "Count"]
            
            fig = px.pie(
                sentiment_dist, 
                names="Sentiment",
                values="Count",
                title="Overall Sentiment Distribution", 
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            category_sentiment = (
                filtered_reviews.groupby(["category", "sentiment_label"]).size().reset_index(name="count")
            )
            
            fig = px.bar(
                category_sentiment, 
                x="category",
                y="count",
                color="sentiment_label",
                title="Category-wise Sentiment Comparison",
                barmode="group"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            
            
        # sentiment trend over time 

        st.subheader("Sentiment Trend Over Time")


        sentiment_trand= (
            filtered_reviews.groupby([pd.Grouper(key="review_date", freq="W"), "sentiment_label"])
            .size()
            .reset_index(name="count")
        )    


        fig_trend = px.line(
            sentiment_trand,
            x="review_date",
            y="count",
            color = "sentiment_label",
            title="Weekly Sentiment Trend"
        )

        st.plotly_chart(fig_trend, use_container_width=True)


        # category trend over time 
        st.subheader("Category Trend Over Time (Product Demand)")

        category_trend=(
            filtered_reviews.groupby([pd.Grouper(key="review_date", freq="M"), "category"])
            .size()
            .reset_index(name="count")
        )

        fig_category_trend = px.line(
            category_trend,
            x="review_date",
            y="count",
            color="category",
            title="Monthly Category Demand Trend"
        )

        st.plotly_chart(fig_category_trend, use_container_width=True)


        # category vs sentiment 
        # Category-wise Sentiment Distribution

        st.subheader("Category-wise Sentiment Distribution")

        cat_sent=(
            filtered_reviews
            .groupby(["category", "sentiment_label"])
            .size()
            .reset_index(name="count")
        )

        fig_cat = px.bar(
            cat_sent,
            x="category",
            y="count",
            color="sentiment_label",
            barmode="group"
        )

        st.plotly_chart(fig_cat, use_container_width=True)


        # topic distribution


        st.subheader("Topic Insights")

        topic_dist = (
            filtered_reviews["topic_label"]
            .value_counts()
            .reset_index()
        )

        topic_dist.columns=["Topic", "Count"]

        fig_topic = px.bar(
            topic_dist, 
            x="Topic",
            y="Count",
            title="Topic Distribution"
        )

        st.plotly_chart(fig_topic, use_container_width=True)


        # Reddit category trend 

        st.subheader("Reddit Category Popularity")

        reddit_trend=(
            reddit_df
            .groupby("category_label")
            .size()
            .reset_index(name="mentions")
            .sort_values("mentions", ascending=False)
        )

        fig_reddit = px.bar(
            reddit_trend, 
            x="category_label",
            y="mentions",
            title="Trending categories on Reddit"
        )

        st.plotly_chart(fig_reddit, use_container_width=True)

        # news sentiment 

        st.subheader("News Sentiment Overview")

        news_sent=(
            news_df
            .groupby("sentiment_label")
            .size()
            .reset_index(name="count")
        )

        fig_news=px.pie(
            news_sent,
            names="sentiment_label",
            values="count",
            title="News Sentiment distribution"
        )

        st.plotly_chart(fig_news, use_container_width=True)


        # News Category Distribution
        # home work



        # trending on all platforms
        # Home work: how we can make it better
        st.subheader("Cross-Source Category Camparision")
        # review category count

        review_cat=(
            reviews_df
            .groupby("category")
            .size()
            .reset_index(name="Review Mentions")
        )

        # reddit category count

        reddit_cat=(
            reddit_df
            .groupby("category_label")
            .size()
            .reset_index(name="Reddit Mentions")
            .rename(columns={"category_label":"category"})
        )

        # news category count

        news_cat=(
            news_df
            .groupby("category")
            .size()
            .reset_index(name="News Mentions")
        )


        # merge all 

        category_compare = review_cat\
            .merge(reddit_cat, on="category", how="outer")\
            .merge(news_cat, on="category", how="outer")\
            .fillna(0)
            
        fig_compare = px.bar(
            category_compare, 
            x="category",
            y=["Review Mentions", "Reddit Mentions", "News Mentions"],
            title="Category Presence Across Reviews, Reddit and News",
            barmode="group"
        )

        st.plotly_chart(fig_compare, use_container_width=True)

        st.dataframe(category_compare)
        
    with right_sidebar:
        st.markdown("## 🤖  AI Insight Panel")    
        st.caption("Ask Question using reviews, news, & Reddit data")
        
        user_query= st.text_area(
            "Your Question",
            height=140
        )
        
        ask_btn = st.button("Get Insight", use_container_width=True)
        
        if ask_btn and user_query:
            with st.spinner("Analyzing Market Intelligence..."):
            
                results = vector_db.similarity_search(user_query, k=10)
                retrived_docs = [r.page_content for r in results]
                
                
                prompt=f"""

                You are a market intelligence analyst
                
                using only the information from the provided context
                
                give response based on the question
                
                do not use bullet points, headings, or sections
                do not add external knowledge
                
                Context:
                {retrived_docs}
                
                Question:
                {user_query}
                
                Answer:
            """   
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                        
                        temperature=0.2
                    ),
                )
            
            st.success("Insight Generated")
            st.write(response.text)
# groq code#


# Load .env so environment variables are available
load_dotenv()
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How does google work with different data sets",
        }
    ],
    model="llama-3.3-70b-versatile",
)
print(chat_completion.choices[0].message.content)