#import tweepy as tw
import streamlit as st
import pandas as pd
from transformers import pipeline



# consumer_key = 'type your API key here'
# consumer_secret = 'type your API key secret here'
# access_token = 'type your Access token here'
# access_token_secret = 'type your Access token secret here'
# auth = tw.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
# api = tw.API(auth, wait_on_rate_limit=True)


classifier = pipeline('sentiment-analysis')

st.title('Live Twitter Sentiment Analysis with Tweepy and HuggingFace Transformers')
st.markdown('This app uses tweepy to get tweets from twitter based on the input name/phrase. It then processes the tweets through HuggingFace transformers pipeline function for sentiment analysis. The resulting sentiments and corresponding tweets are then put in a dataframe for display which is what you see as result.')
#     search_words = st.text_input('Enter the name for which you want to know the sentiment')
    
def run():
    with st.form(key='Enter name'):
        st.header('Upload the twitter data for sentiment analysis')
        uploaded_file = st.file_uploader('Upload a file')
        number_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment(Maximum 50 tweets)', 0,50,10)
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            #tweets =tw.Cursor(api.search_tweets,q=search_words,lang=”en”).items(number_of_tweets)
            #tweet_list = [i.text for i in tweets]
            tweets = pd.read_csv(uploaded_file)
            final_tweets = tweets[['text']].head(number_of_tweets)
            tweet_list = final_tweets['text'].tolist()
            p = [i for i in classifier(tweet_list)]
            q=[p[i]['label'] for i in range(len(p))]
            df = pd.DataFrame(list(zip(tweet_list, q)),columns =['Latest'+str(number_of_tweets)+'Tweets'+'on'+"Customer Support on Twitter", 'sentiment'])
            chart_data = df['sentiment'].value_counts().reset_index()
            st.write(df)
            st.bar_chart(chart_data)
 

if __name__=='__main__':
    run()
