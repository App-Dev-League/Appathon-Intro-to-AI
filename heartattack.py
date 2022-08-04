import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("heart.csv") # Note how few examples there are, only 303. This is not enough for larger models
# such as neural networks

data = data.sample(frac=1, random_state=1).reset_index().iloc[:, 1:] # randomly shuffles data (could also pass in seed)
# the indexing removes the first column which is called "index" and is not needed (gen. by the sample func)

#Create X and y

X = data.iloc[:, :13] # Not all of these columns will be needed, will slice later
print(X.columns) # shows column names for the dataframe
print(X.shape) # prints the dataframe shape to confirm proper slicing

y = data.iloc[:, 13]
print(y.name) # prints the name of the vector
print(y.shape) # prints the vector's length (ie. shape)


# Now let's quickly partition the data into a train and test split (242 is 80% of the data)
trainX = X[:242]
trainy = y[:242]
testX = X[242:]
testy = y[242:]
# This is fine most of the time, but you can also use sklearns builtin functions for better performance on large datasets


# LOGISTIC REGRESSION #
# For this, we will use only one column of data
# Let us choose age as a simple regressor (though this may not be the best describing variable)

trainAge = trainX.age.values.reshape(-1, 1) # the reshape allows the model to work with only 1 feature (it is built for matrices)
testAge = testX.age.values.reshape(-1, 1)
model = LogisticRegression().fit(trainAge, trainy)

model.predict(testAge) # could print this to see actual values
print(model.score(testAge, testy)) # should be about 0.67

# Now let's try using all of the features!
model = LogisticRegression().fit(trainX, trainy)

model.predict(testX)
print(model.score(testX, testy)) # should be about 0.93

# Look at how big of a difference including more features makes this model!
# In this case it is unnecessary, but in the future you will need better models too!
# Keep experimenting and enjoy!
