from typing import List, Iterable, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import numpy as np
import scipy as sp
import nltk
import json
import pickle


class Solution:
    @staticmethod
    def download_stopwords():
        nltk.download('stopwords')
        words = list(stopwords.words("russian"))
        out_f = open("./stopwords.bin", "wb")
        pickle.dump(words, out_f)

    def __init__(self):
        in_f = open("./stopwords.bin", "rb")
        words = pickle.load(in_f)

        dataset = json.load(open("./dev-dataset-task2022-04.json", "r"))
        x, y = [d[0] for d in dataset], [int(d[1]) for d in dataset]
        x, y = np.array(x), np.array(y)

        self.vect = TfidfVectorizer(
            input="content",
            analyzer="word",
            stop_words=words
        )
        x = self.vect.fit_transform(x)

        self.model_cls = KNeighborsClassifier
        self.experiment_model_kwargs = [
            {
                "n_neighbors": 3,
                "metric": "euclidean",
                "weights": "uniform"
            },
            {
                "n_neighbors": 3,
                "metric": "cosine",
                "weights": "uniform"
            },
            {
                "n_neighbors": 3,
                "metric": "euclidean",
                "weights": "distance"
            },
            {
                "n_neighbors": 3,
                "metric": "cosine",
                "weights": "distance"
            },
            {
                "n_neighbors": 5,
                "metric": "euclidean",
                "weights": "uniform"
            },
            {
                "n_neighbors": 5,
                "metric": "cosine",
                "weights": "uniform"
            },
            {
                "n_neighbors": 5,
                "metric": "euclidean",
                "weights": "distance"
            },
            {
                "n_neighbors": 5,
                "metric": "cosine",
                "weights": "distance"
            },
            {
                "n_neighbors": 7,
                "metric": "euclidean",
                "weights": "uniform"
            },
            {
                "n_neighbors": 7,
                "metric": "cosine",
                "weights": "uniform"
            },
            {
                "n_neighbors": 7,
                "metric": "euclidean",
                "weights": "distance"
            },
            {
                "n_neighbors": 7,
                "metric": "cosine",
                "weights": "distance"
            },
        ]

        self.x, self.y = x, y
        self.learn_each_steps = 1
        self.predict_model_kwargs = {
            "n_neighbors": 5,
            "metric": "cosine",
            "weights": "distance"
        }

        self.predict_model = self.model_cls(**self.predict_model_kwargs)
        self.predict_model.fit(self.x, self.y)
        self.step = 0

    def experiment(self):
        dataset = json.load(open("./dev-dataset-task2022-04.json", "r"))
        x, y = [d[0] for d in dataset], [int(d[1]) for d in dataset]

        x = self.vect.transform(np.array(x))
        y = np.array(y)

        scores = dict()
        for kwargs in self.experiment_model_kwargs:
            model = self.model_cls(**kwargs)
            scores[tuple(kwargs.values())] = np.mean(cross_val_score(model, x, y, cv=3))

        print("Cross validation scores:")
        for k, v in scores.items():
            print(k, v)
        print()

        best_kwargs = max(scores, key=scores.get)
        best_kwargs = {
            "n_neighbors": best_kwargs[0],
            "metric": best_kwargs[1],
            "weights": best_kwargs[2]
        }

        print("Best model parameters:")
        print(best_kwargs)
        print()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, shuffle=True)
        model = self.model_cls(**best_kwargs)
        model.fit(x_train, y_train)

        print("Best model test score:")
        print(f1_score(y_test, model.predict(x_test), average="macro"))
        print()

    def predict(self, text: str) -> str:
        if self.step == self.learn_each_steps:
            self.predict_model = self.model_cls(**self.predict_model_kwargs)
            self.predict_model.fit(self.x, self.y)
            self.step = 0

        x = np.array([text])
        x = self.vect.transform(x)
        y = self.predict_model.predict(x)

        self.x = sp.sparse.vstack((self.x, x), format='csr')
        self.y = np.append(self.y, y, axis=0)
        self.step += 1

        return str(y[0])


if __name__ == "__main__":
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    Solution.download_stopwords()
    sol = Solution()
    sol.experiment()

    dataset = json.load(open("./dev-dataset-task2022-04.json", "r"))
    x, y = [d[0] for d in dataset], [int(d[1]) for d in dataset]

    for i in range(100):
        pred = sol.predict(x[i])
        print(pred, y[i])
