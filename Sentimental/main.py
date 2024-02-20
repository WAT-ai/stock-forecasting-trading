from senti import SentimentClassifier

from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key="3cad96ff396240f1afccd80f57e87201")

def get_news_sentiment(company, classifier):
    news_sentiment = []
    top_headlines = newsapi.get_top_headlines(q=company,
                                          category='business',
                                          language='en',
                                          country='us')
    for i in range(top_headlines["totalResults"]):
        cur_article = top_headlines["articles"][i]
        news_sentiment.append(classifier.get_sentiment(cur_article["title"]))
    
    return news_sentiment



def main():
    csv_path = input("Enter CSV Path: ")
    ticker = input("Enter ticker or company name: ")
    sent_class = SentimentClassifier(csv_path)
    # example = sent_class.get_sentiment("NVIDIA forecasts increased demand for their chips, sales up 50%")
    # print(example)
    news_sentiment = get_news_sentiment(ticker, sent_class)
    print(news_sentiment)

if __name__ == '__main__':
    print("Running...")
    main()