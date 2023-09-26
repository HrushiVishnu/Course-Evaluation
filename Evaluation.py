import streamlit as st
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import pandas as pd
import numpy as np
import re
from transformers import pipeline
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#summarizer = pipeline('summarization')

def get_pdf_text(pdf_docs):
    text = ""
    separator = "Comments on Course Characteristics"
    found_separator = False 
    
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            
            if found_separator:
                text += page_text
                continue
            
            if separator in page_text:
                text += page_text.split(separator, 1)[1]
                found_separator = True
    
    return text

def extract_comments(text):
    course_comments_segment, instructor_text = text.split("Comments on Instructor Characteristics", 1)
    course_comments = course_comments_segment.split("___________________________________________________________________")[1:]
    instructor_comments = instructor_text.split("___________________________________________________________________")[1:]

    def clean_comment(comment):
        comment = re.sub(r'Page\d+', '', comment)
        comment = re.sub(r'Kenan-Flagler Course Evaluation Report', '', comment)
        return comment.strip()

    course_comments = [clean_comment(comment) for comment in course_comments]
    instructor_comments = [clean_comment(comment) for comment in instructor_comments]

    df_course_comments = pd.DataFrame(course_comments, columns=["Course Comments"])
    df_instructor_comments = pd.DataFrame(instructor_comments, columns=["Instructor Comments"])

    return df_course_comments, df_instructor_comments


def create_wordcloud(text):
    wordcloud = WordCloud(background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()


def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    if sentiment_scores['compound'] > 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] < -0.05:
        return 'Negative'
    else:
        return 'Neutral'
    
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')
        
def plot_sentiment_distribution(df, sentiment_column, column_name):
    sentiment_counts = df[sentiment_column].value_counts()
    
    ax = sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'], figsize=(10, 6))
    
    ax.set_ylim(0, max(sentiment_counts) * 1.1)  
    
    ax.set_yticks(np.arange(0, max(sentiment_counts) * 1.1, 1)) 
    
    autolabel(ax.patches, ax)
    
    plt.title(f"Sentiment Distribution for {column_name}")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Comments")
    st.pyplot()


# def get_summary(text):
#     if len(text.split()) <= 100:  
#         return "Comments is too short to summarize."

#     summary_result = summarizer(text, min_length=30, max_length=50)
#     return summary_result[0]['summary_text']


def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(layout="wide")
    with st.sidebar:
        image = Image.open('UI/AI_Lab_Logo.png')
        st.image(image)
    st.title("Course Evaluation Dashboard")

    pdf_docs = st.file_uploader(
        "Upload your Course Evaluation Report here and click on 'Process'", accept_multiple_files=True
    )

    text = ""

    if st.button("Process"):
        text = get_pdf_text(pdf_docs)
        df_course_comments, df_instructor_comments = extract_comments(text)

        df_course_comments['Sentiment'] = df_course_comments['Course Comments'].apply(get_sentiment)
        df_instructor_comments['Sentiment'] = df_instructor_comments['Instructor Comments'].apply(get_sentiment)

        # df_course_comments['Summary'] = df_course_comments['Course Comments'].apply(get_summary)
        # df_instructor_comments['Summary'] = df_instructor_comments['Instructor Comments'].apply(get_summary)

        total_comments = len(df_course_comments) + len(df_instructor_comments)
        combined_df = pd.concat([df_course_comments, df_instructor_comments])
        sentiment_counts = combined_df['Sentiment'].value_counts()
        sentiment_percentages = (sentiment_counts / total_comments) * 100

        colors = {
            "Positive": "green",
            "Negative": "red",
            "Neutral": "blue"
        }

        sentiment_display = " &nbsp; ".join([f'<span style="color: {colors[sentiment]}">{sentiment}: {percentage:.2f}%</span>' for sentiment, percentage in sentiment_percentages.items()])
        st.markdown(f"**Overall Sentiment Percentages:**\n\n{sentiment_display}", unsafe_allow_html=True)

        st.markdown("# Course Comments:")
        st.write(df_course_comments)
        plot_sentiment_distribution(df_course_comments, 'Sentiment', 'Course Comments')
        course_text_for_wordcloud = ' '.join(df_course_comments['Course Comments'])
        st.subheader("Word Cloud for Course Comments")
        create_wordcloud(course_text_for_wordcloud)

        st.markdown("# Instructor Comments")
        st.write(df_instructor_comments)
        plot_sentiment_distribution(df_instructor_comments, 'Sentiment', 'Instructor Comments')
        instructor_text_for_wordcloud = ' '.join(df_instructor_comments['Instructor Comments'])
        st.subheader("Word Cloud for Instructor Comments")
        create_wordcloud(instructor_text_for_wordcloud)

if __name__ == '__main__':
    main()
