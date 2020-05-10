import pandas as pd
from numpy.random import permutation
from sklearn import preprocessing as ppr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.neighbors import KNeighborsClassifier as knc

with open('479Proj2.csv') as covidfile:
    cases = pd.read_csv(covidfile, index_col="id")

#preprocess data (normalize)
def normalizeDF(df, colNames):
    scaler = ppr.MinMaxScaler()
    dfCopy = df
    dfCopy[colNames] = scaler.fit_transform(df[features])
    return dfCopy


features = ['age', 'bmi', 'HbA1c']
cases = normalizeDF(cases, features)
features.append('resp_disease')
X = cases[features].values
y = cases['death_risk'].values

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#knn classification
classifier = knc(n_neighbors=4)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(y_pred)
accuracy = acc(y_test, y_pred)*100
print(accuracy)
'''

random_indices = permutation(cases.index)
test_cutoff = math.floor(len(cases) / 3)
test = cases.loc[random_indices[1:test_cutoff]]
train = cases.loc[random_indices[test_cutoff:]]

knn = knc(n_neighbors=5)
knn.fit(train[X], train[y])
predictions = knn.predict(test[X])


'''


