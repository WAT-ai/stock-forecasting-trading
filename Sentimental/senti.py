import csv
import kaggle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SentimentClassifier:
    def __init__(self, csv_path):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files("ankurzing/sentiment-analysis-for-financial-news", path=csv_path, unzip=True)
        self.glove = torchtext.vocab.GloVe(name="6B", dim=300)
        self.train, self.valid = self._compute_tweet_glove_vectors(self.glove)
        self.train_loader = torch.utils.data.DataLoader(self.train, batch_size=128, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(self.valid, batch_size=128, shuffle=True)
        self.model = nn.Sequential(nn.Linear(300, 100),
                        nn.BatchNorm1d(100),
                        nn.ReLU(),
                        nn.Dropout(0.25),
                        nn.Linear(100, 80),
                        nn.BatchNorm1d(80),
                        nn.ReLU(),
                        nn.Dropout(0.25),
                        nn.Linear(80, 40),
                        nn.BatchNorm1d(40),
                        nn.ReLU(),
                        nn.Dropout(0.25),
                        nn.Linear(40, 10),
                        nn.BatchNorm1d(10),
                        nn.ReLU(),
                        nn.Linear(10, 2),
                        nn.Softmax(dim=1))
        self._train_nn(self.model, self.train_loader, self.valid_loader, num_epochs=500, learning_rate=1e-4, show_plots = False)
        self.model.eval()


    def _get_data(self, file_name):
        return csv.reader(open(file_name, "rt", encoding="latin-1"))
    
    def _split_line(self, line):
        line = line.replace(".", " . ") \
                    .replace(",", " , ") \
                    .replace(";", " ; ") \
                    .replace("?", " ? ")
        return line.split()

    def _create_glove_embedding(self, line, glove_vector):
        return sum(glove_vector[w] for w in self._split_line(line))
    
    def _compute_tweet_glove_vectors(self, glove_vector):
        train, valid = [], []
        for i, line in enumerate(self._get_data("all-data.csv")):
            tweet = line[1]
            if line[0] == "neutral":
                continue
            vector_sum = self._create_glove_embedding(tweet, glove_vector)
            label = torch.tensor(1 if line[0] == "positive" else 0 ).long()
            if i % 5 <= 3:
                train.append((vector_sum, label))
            else:
                valid.append((vector_sum, label))

        return train, valid

    def _get_accuracy(self, model, data_loader):
        correct, total = 0, 0
        for tweets, labels in data_loader:
            output = model(tweets)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.shape[0]
        return correct / total
    
    def _train_nn(self, model, train_loader, valid_loader, num_epochs=5, learning_rate=1e-5, show_plots=False):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        losses, train_accuracy, val_accuracy = [], [], []
        epochs = []
        for epoch in range(num_epochs):
            print(epoch)
            for tweets, labels in train_loader:
                optimizer.zero_grad()
                pred = model(tweets)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
            losses.append(float(loss))

            if epoch % 5 == 4:
                epochs.append(epoch)
                train_accuracy.append(self._get_accuracy(model, train_loader))
                val_accuracy.append(self._get_accuracy(model, valid_loader))
                print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
                    epoch+1, loss, train_accuracy[-1], val_accuracy[-1]))

        if show_plots:
            plt.title("Training Curve")
            plt.plot(losses, label="Train")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()

            plt.title("Training Curve")
            plt.plot(epochs, train_accuracy, label="Train")
            plt.plot(epochs, val_accuracy, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(loc='best')
            plt.show()

    
    def get_sentiment(self, query):
        embedding = self._create_glove_embedding(query, self.glove)
        embedding = embedding.unsqueeze(0)
        
        out = self.model(embedding)
        
        class_probabilities = torch.softmax(out, dim=1)
        
        class_prob, topclass = torch.max(class_probabilities, dim=1)

        return topclass.item(), class_prob.item()   