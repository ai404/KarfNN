import numpy as np
import pandas

from KarfNN.layer import Dense, Dropout
from KarfNN.models import Karf

from datasets import load

def toDummies(df, Columns):
    for Column in Columns:
        new_df = pandas.get_dummies(df[Column], prefix=Column)
        df = pandas.concat([df, new_df], axis=1)
    df = df.drop(Columns, axis=1)
    return df

np.random.seed(1)

df = load("Iris")

# shuffle data
data = df.iloc[np.random.permutation(len(df))]

# split data to X and y and code Species names to numbers
X = data.drop(["Id", "Species"], axis=1).astype(np.float)
y = data[["Species"]]

# OneHot encoding for output vector
y = toDummies(y,["Species"])

# split data to training sets and testing sets
train_split = int(len(X) * 0.75)

Xtrain = X[:train_split].values
ytrain = y[:train_split].values
Xtest = X[train_split:].values
ytest = y[train_split:].values

# initialize the model
model = Karf()

model.init(Xtrain, ytrain)
model.add(Dense(n_units=10,activation="relu"))
model.add(Dropout(threshold=0.9))
model.add(Dense(n_units=10,activation="relu"))
model.add(Dense(n_units=10,activation="relu"))
model.add(Dropout(threshold=0.9))
model.add(Dense(n_units=3,activation="softmax"))

model.run(n_epochs=15000,batch_size=40,learning_rate=1e-4,optimizer="adam",loss="cross_entropy")

prediction = model.predict(Xtest).argmax(axis=1)
score = np.sum(prediction == ytest.argmax(axis=1))*1./len(prediction)*100
print ("score: %.2f" % score+"%")