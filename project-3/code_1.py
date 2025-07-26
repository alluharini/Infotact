import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

df = pd.read_excel('Tweets.xlsx') 

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].astype(str).apply(preprocess)

# Sentiment Classification using TextBlob
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['predicted_sentiment'] = df['clean_text'].apply(get_sentiment)

# Save processed data
df.to_csv('processed_tweets.csv', index=False)

# WordCloud for each sentiment
sentiments = ['positive', 'neutral', 'negative']
plt.figure(figsize=(20, 10))

for i, sentiment in enumerate(sentiments):
    text = ' '.join(df[df['predicted_sentiment'] == sentiment]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.subplot(1, 3, i + 1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{sentiment.capitalize()} Sentiment')

plt.tight_layout()
plt.show()

# Sentiment Distribution Plot
sns.countplot(data=df, x='predicted_sentiment', palette='pastel')
plt.title('Tweet Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()