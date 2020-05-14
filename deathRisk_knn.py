from mpi4py import MPI
import os
import pandas as pd
from sklearn import preprocessing as ppr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.neighbors import KNeighborsClassifier as knc

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

def covid_knn(trainfile, testfile, process_rank):
    with open(trainfile) as covidfile:
        cases = pd.read_csv(covidfile, index_col="id")

    with open(testfile) as casefile:
        tests = pd.read_csv(casefile, index_col="id")

    features = ['age', 'bmi', 'HbA1c']
    cases = normalizeDF(cases, features)

    features.append('resp_disease')
    X = cases[features].values
    y = cases['death_risk'].values

    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # knn classification
    classifier = knc(n_neighbors=6)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # knn testing with input files
    tests = normalizeDF(tests, features)
    Z = tests[features].values
    Z_pred = classifier.predict(Z)
    print("The predictions for: " + testfile)
    print(Z_pred)

    if process_rank == 0:
        accuracy = acc(y_test, y_pred) * 100
        print("Accuracy of the model is: ")
        print(accuracy)


# preprocess data (normalize)
def normalizeDF(df, colNames):
    scaler = ppr.MinMaxScaler()
    dfCopy = df
    dfCopy[colNames] = scaler.fit_transform(df[colNames])
    return dfCopy


def getcsv(ignorefile=""):
    filelist = []
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(".csv") and filename != ignorefile:
            filelist.append(filename)
    return filelist


filelist = getcsv(ignorefile="479Proj2.csv")

covid_knn("479Proj2.csv", filelist[rank], rank)

MPI.Finalize()