from senti import SentimentClassifier


def main():
    csv_path = input("Enter CSV Path: ")
    sent_class = SentimentClassifier(csv_path)
    example = sent_class.get_sentiment("NVIDIA forecasts increased demand for their chips, sales up 50%")
    print(example)

if __name__ == '__main__':
    print("Running...")
    main()