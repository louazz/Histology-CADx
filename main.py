import pickle

import itertools
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay, \
    ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.TorchDataset import DatasetTorch
import numpy as np
import matplotlib.pyplot as plt

class TrainModel:
    def __init__(self):
        self.pca = PCA(n_components=0.9)
        self.svm = SVC(kernel="rbf", probability=True)
        self.rf = RandomForestClassifier()
        self.writer = SummaryWriter()
        self.nn_model = torch.nn.Sequential(
            torch.nn.Linear(26, 13),
            torch.nn.BatchNorm1d(13),
            torch.nn.ELU(),

            torch.nn.Linear(13, 6),
            torch.nn.BatchNorm1d(6),
            torch.nn.ELU(),

            # torch.nn.Dropout(0.2),

            torch.nn.Linear(6, 1),
            torch.nn.Sigmoid()
        )
        self.data = DatasetTorch()

    def train_neural_network(self):
            ds = self.data
            ds.read_data()

            kf = KFold(n_splits=5, shuffle=True)
            n_epochs = 300
            loss_values = []

            true_val = []
            pred_val = []

            j = 0

            for train_idx, test_idx in kf.split(ds):
                j += 1
                optim = torch.optim.Adam(self.nn_model.parameters(), lr=1e-4)
                loss_fn = torch.nn.BCELoss()

                train_set = [ds[i] for i in train_idx]
                test_set = [ds[i] for i in test_idx]

                train_loader = DataLoader(dataset=train_set, batch_size=6, shuffle=True)
                test_loader = DataLoader(dataset=test_set, batch_size=6, shuffle=True)

                i = 0
                running_loss = 0.0
                for epoch in range(n_epochs):
                    i += 1
                    print('Epoch number: {}'.format(i))

                    k = 0
                    for X, y in train_loader:
                        k += 1
                        optim.zero_grad()

                        pred = self.nn_model(X)
                        loss = loss_fn(pred, y.to(torch.float32).unsqueeze(1))

                        loss_values.append(loss.item())
                        loss.backward()
                        optim.step()
                        running_loss += loss.item()
                        self.writer.add_scalars('run_split_{}'.format(j),
                                                {'training loss': running_loss},
                                                epoch)

                print('Completed training')

                y_true = []
                y_pred = []

                total = 0.0
                correct = 0.0

                with torch.no_grad():
                    for X, y in test_loader:
                        print(X.shape)
                        out = self.nn_model(X)
                        pred = np.where(out < 0.5, 0, 1)
                        pred = list(itertools.chain(*pred))
                        y_pred.extend(pred)
                        y_true.extend(y.data.cpu().numpy())
                        total += y.size(0)
                        correct += (pred == y.numpy().sum().item())

                report = classification_report(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred)

                print(report)
                print(cm)

                true_val.extend(y_true)
                pred_val.extend(y_pred)
            print('-' * 10)
            report = classification_report(true_val, pred_val)
            print(report)
            cm = confusion_matrix(true_val, pred_val)
            print(cm)
            fpr, tpr, _ = roc_curve(true_val, pred_val)
            print(tpr)
            print(fpr)
            roc_auc = roc_auc_score(true_val, pred_val)
            print(roc_auc)
            roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
            roc_display.plot()

            display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(['benign', 'malignant']))
            display.plot()
            plt.show()

    def cross_val_train(self):
            with open("ds/data.pkl", "rb") as f:
                ds = pickle.load(f)

            kf = KFold(n_splits=5, shuffle=True)

            true = np.array([])
            pred = np.array([])
            proba = []

            sampler = RandomUnderSampler(sampling_strategy="majority")

            for train_idx, test_idx in kf.split(ds):
                print(ds)
                train_set =[ds[i] for i in train_idx]
                test_set = [ds[i] for i in test_idx]

                y_train = []
                desc_train = []
                for data in train_set:
                    y_train.append(data.label)
                    desc_train.append(data.descriptor)

                desc_train, y_train = sampler.fit_resample(desc_train, y_train)
                y_test = []
                desc_test = []
                for data in test_set:
                    y_test.append(data.label)
                    desc_test.append(data.descriptor)

                self.pca.fit(desc_train, y_train)

                reduced_train = self.pca.transform(desc_train)
                reduced_test = self.pca.transform(desc_test)

                self.svm.fit(reduced_train, y_train)

                y_pred = self.svm.predict(reduced_test)
                prob = self.svm.predict_proba(reduced_test)

                pred = np.concatenate((pred, y_pred), axis=0)
                true = np.concatenate((true, y_test), axis=0)
                proba.extend(prob[:, 1])

                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred)

                print("The confusion matrix of the RF model is")
                print(cm)
                print(report)

            print("-" * 10)

            cm = confusion_matrix(true, pred)
            report = classification_report(true, pred)

            print("The confusion matrix of the RF model is")
            print(cm)
            print(report)
            fpr, tpr, _ = roc_curve(true, pred)
            print(tpr)
            print(fpr)
            roc_auc = roc_auc_score(true, pred)
            print(roc_auc)
            roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
            roc_display.plot()

            display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(['benign', 'malignant']))
            display.plot()

            plt.show()


if __name__ == "__main__":
    lbp = TrainModel()
    lbp.cross_val_train()
