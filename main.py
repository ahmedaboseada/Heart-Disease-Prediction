import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('heart_disease_dataset.csv')

print(dataset)

text_columns = ['Gender', 'Smoking', 'Alcohol Intake', 'Chest Pain Type', 'Family History', 'Diabetes', 'Obesity',
                'Exercise Induced Angina']

## convert string data to numerical
label_encoder = LabelEncoder()

print(dataset.shape)

for col in text_columns:
    dataset[col] = label_encoder.fit_transform(dataset[col])

print(dataset)

plt.figure(figsize=(12, 12))
correlation = dataset.corr()
sns.heatmap(correlation, annot=True, cmap=plt.cm.Reds)
plt.show()

data = dataset.drop(columns='Heart Disease')
target = dataset['Heart Disease']
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3, random_state=1)

###
# training and testing the model
###

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(data_train)
# data_train = scaler.transform(data_train)
# data_test = scaler.transform(data_test)

# from sklearn.decomposition import PCA
# pca = PCA()
#
# data_train = pca.fit_transform(data_train)
# data_test = pca.transform(data_test)

print(data_train.std())

# import warnings
# warnings.filterwarnings('ignore', message="X has feature names")

classifier = svm.NuSVC(nu=0.078)  # 99.14%
classifier.fit(data_train, target_train)
data_train_predict = classifier.predict(data_train)
train_predictions = accuracy_score(data_train_predict, target_train)
data_test_predict = classifier.predict(data_test)
test_predictions = accuracy_score(data_test_predict, target_test)
print("Test Accuracy:", round(test_predictions * 100, 2), "%")
print("Train Accuracy:", round(train_predictions * 100, 2), "%")
print(dataset.shape)

###
# Testing
###

testing1 = [48, 1, 204, 165, 62, 0, 2, 5, 0, 0, 0, 9, 70, 1, 3]  # 0
testing2 = [60, 1, 226, 168, 99, 2, 1, 8, 1, 1, 0, 10, 97, 0, 2]  # 1
testing3 = [20, 1, 224, 119, 79, 0, 0, 12, 1, 1, 1, 9, 173, 1, 1]  # ?

testing_df1 = pd.DataFrame([testing1], columns=data.columns)
prediction1 = classifier.predict(testing_df1)
print(prediction1)
testing_df2 = pd.DataFrame([testing2], columns=data.columns)
prediction2 = classifier.predict(testing_df2)
print(prediction2)
testing_df3 = pd.DataFrame([testing3], columns=data.columns)
prediction3 = classifier.predict(testing_df3)
print(prediction3)
