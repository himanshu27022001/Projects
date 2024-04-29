# Machine learning Project
#Water Potability detection
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("/content/drive/MyDrive/water_potability.csv") # reading dataset
from google.colab import drive
drive.mount('/content/drive')
df.head() # display first five rows
df.shape # the shape of our dataset
df.dtypes
df.hist(figsize = (20,10), layout = (3,4))
plt.show()
df.isnull().sum() # total number of NaN values in all columns
# Replace NaN values with medians of those columns
df['ph'] = df['ph'].fillna(df['ph'].median())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].median())
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].median())
df.isnull().sum()
values = df['Potability'].value_counts().to_list()
labels = df['Potability'].value_counts().index.to_list()
plt.pie(values, labels = labels, autopct = "%1.1f%%", explode = [0.05, 0.05], shadow = True, startangle = 120)
plt.show()
fig = plt.figure()
fig.suptitle("Distribution Plots", fontsize = 25)
fig.subplots_adjust(wspace = 0.2, hspace = 0.3)
for i,x in enumerate(df.columns):
    ax = fig.add_subplot(4,3,i+1)
    fig.set_figheight(20)
    fig.set_figwidth(20)
    sns.distplot(df[x], hist = False, color = 'violet', kde_kws = {'shade': True})
plt.show()
df['Solids'] = np.power(df['Solids'], 1/2)
sns.distplot(df['Solids'], hist = False, color = 'violet', kde_kws = {'shade' : True})
fig = plt.figure()
fig.suptitle("Violin Plots", fontsize = 25)
fig.subplots_adjust(wspace = 0.2, hspace = 0.3)
for i,x in enumerate(df.columns):
    ax = fig.add_subplot(4,3,i+1)
    fig.set_figheight(20)
    fig.set_figwidth(20)
    sns.violinplot(x = df['Potability'], y = df[x])
plt.show()
sns.pairplot(df, hue = "Potability")
plt.figure(figsize = (10,10))
sns.heatmap(df.corr(), annot = True, cmap = "RdYlGn")

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(xtrain, ytrain)
xtrain_scaled = scale.transform(xtrain)
xtest_scaled = scale.transform(xtest)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(xtrain_scaled, ytrain)
yhat_logreg = logreg.predict(xtest_scaled)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(yhat_logreg, ytest), confusion_matrix(yhat_logreg, ytest),
      classification_report(yhat_logreg, ytest), sep = '\n\n')
      
      from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(xtrain_scaled, ytrain)
yhat_knn = knn.predict(xtest_scaled)

print(accuracy_score(yhat_knn, ytest), confusion_matrix(yhat_knn, ytest),
      classification_report(yhat_knn, ytest), sep = '\n\n')
      
      train_score = []
test_score = []
for n in range(2,20,2):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(xtrain_scaled, ytrain)
    train_score.append(knn.score(xtrain_scaled, ytrain))
    test_score.append(knn.score(xtest_scaled, ytest))
plt.plot(train_score, color = 'r', label = 'train score')
plt.plot(test_score,color = 'g', label = 'test_score')
plt.legend()

knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(xtrain_scaled, ytrain)
yhat_knn = knn.predict(xtest_scaled)
print(accuracy_score(yhat_logreg, ytest), confusion_matrix(yhat_logreg, ytest),
      classification_report(yhat_logreg, ytest), sep = '\n\n')
      knn_score = accuracy_score(yhat_knn, ytest)
