#%%
# Read dataset

import pandas as pd

train = pd.read_csv('dataset/train.csv')

print(train.head())
#%%
# Print graph

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(train['first'], train['label'])
plt.xlabel('first')
plt.ylabel('label')
plt.show()
#%%
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

model = Pipeline([
    ('poly', PolynomialFeatures(degree = 3)),
    ('scaler', StandardScaler()),
    ('elastic', ElasticNet(random_state=42)),
])
#%%
features = train.drop(columns=['id', 'label'])

print(features.head())
#%%
X = train[features]
y = train['label']
#%%
# Test model

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
#%%
# Run private test

test = pd.read_csv('dataset/test.csv')

y_pred = model.predict(test[features])

submission = pd.DataFrame({
    'id' : test['id'],
    'label' : y_pred
})

submission.to_csv('submission.csv', index=False)