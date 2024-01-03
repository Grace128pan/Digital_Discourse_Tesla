import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load ROBERTA and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']

# Load the CSV file
target_file = "C:\\Users\\grace\\Desktop\\Digital_Discourse\\Merged_Tesla\\20240103_merged_tesla.csv"
# Replace with the path to your CSV file
# and change "\" into "\\"!!!!!!
df = pd.read_csv(target_file)


# Define a function to perform sentiment analysis
def perform_sentiment_analysis(tweet):
    # Check if tweet is NaN or not a string
    if pd.isna(tweet) or not isinstance(tweet, str):
        return 0.0  # Return a default sentiment score (you can modify this as needed)

    tweet_words = []

    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')

    output = model(**encoded_tweet)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Calculate the final sentiment score
    positive_score = scores[2]
    negative_score = scores[0]
    return positive_score - negative_score


# Perform sentiment analysis and add scores to the DataFrame
df['scores'] = df['Tweet Content'].apply(perform_sentiment_analysis)

# Save the DataFrame back to the CSV file
df.to_csv(target_file, index=False)

print("RoBERTa Analysis Complete!")
