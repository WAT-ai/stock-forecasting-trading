#pip install -r requirements.txt

from model import LstmModel

def main():
    ticker = input("Enter ticker name: ")
    model = LstmModel(ticker)
    model.download_data()
    model.preprocess_data()
    model.build_model()
    model.train_model()
    print(model.predict_future()) #model.predict_future() return the pridicted price of the stock after 30 days

if __name__ == '__main__':
    main()
