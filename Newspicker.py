
# news_headline_analyzer.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

st.set_page_config(page_title="Real-time News Headline Analyzer", layout="wide")
st.title("📰 Real-time News Headline Extractor & Analyzer")
st.markdown("Analyzes the latest headlines from [Hacker News](https://news.ycombinator.com) for sentiment and word frequency.")

NEWS_URL = "https://news.ycombinator.com/"

@st.cache_data(show_spinner=True)
def get_latest_headlines(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []

        for title_span in soup.find_all('span', class_='titleline'):
            link_tag = title_span.find('a')
            if link_tag:
                headline_text = link_tag.get_text(strip=True)
                if headline_text:
                    headlines.append(headline_text)

        for td_title in soup.find_all('td', class_='title'):
            link_tag = td_title.find('a')
            if link_tag and link_tag.get_text(strip=True) != 'More':
                headline_text = link_tag.get_text(strip=True)
                if headline_text and headline_text not in headlines:
                    headlines.append(headline_text)

        return list(set(headlines))

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching page: {e}")
        return []

def analyze_headlines(headlines_list):
    all_words = []
    sentiment_results = []

    stopwords = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'to', 'of', 'in', 'on', 'at', 'with', 'from', 'by', 'for', 'about', 'as', 'into', 'through',
        'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'new', 'says', 'say', 'us', 'here'
    ])

    for headline in headlines_list:
        words = re.findall(r'\b\w+\b', headline.lower())
        filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
        all_words.extend(filtered_words)

        analysis = TextBlob(headline)
        polarity = analysis.sentiment.polarity
        sentiment_category = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"

        sentiment_results.append({
            'Headline': headline,
            'Polarity': polarity,
            'Sentiment': sentiment_category
        })

    word_counts = Counter(all_words)
    return word_counts, sentiment_results

# Main App Flow
headlines = get_latest_headlines(NEWS_URL)

if not headlines:
    st.warning("No headlines found or network error.")
else:
    st.success(f"Fetched {len(headlines)} unique headlines.")

    word_frequencies, sentiment_data = analyze_headlines(headlines)

    st.subheader("📊 Sentiment Analysis of Headlines")
    df_sentiment = pd.DataFrame(sentiment_data)
    st.dataframe(df_sentiment[['Headline', 'Sentiment', 'Polarity']], use_container_width=True)

    sentiment_counts = df_sentiment['Sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    st.subheader("🔠 Top 10 Most Frequent Words")
    top_words = word_frequencies.most_common(10)
    df_top_words = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    st.dataframe(df_top_words)

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Frequency', y='Word', data=df_top_words, palette='viridis')
    plt.title("Top 10 Frequent Words in Headlines")
    st.pyplot(plt.gcf())
