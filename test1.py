import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("C:/Users/hp/Desktop/machine learning/breast-cancer-data.csv")
df.head(5)
diagnosis = pd.get_dummies(df.diagnosis)
df = pd.concat([df, diagnosis], axis = 1)
df = df.drop(['diagnosis'], axis=1)
df.rename(columns = {'B' : 0, 'M' : 1})
df=df.drop("id",axis=1)
df=df.drop("B",axis=1)
X=df.drop("M",axis=1)
y=df["M"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=42)
classifier.fit(X_train,y_train)
pickle_out=open("model.pkl","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close()