import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn import metrics
import time

# Import the data from csv:

all_data = pd.read_csv("profiles.csv")

# Explore the data

# I'd like to explore the income feature. What are the best predictors of income?
# print(all_data.income.value_counts())

# Features that traditionally are expected to be relevan for income
# print(all_data.education.value_counts())
# print(all_data.job.value_counts())

# Taller people are usually thought to have more charisma. Are they paid more?

# print(all_data.height.value_counts())

cleaned_height = all_data.height.replace(np.nan, 0, regex=True)
# plt.hist(cleaned_height, bins=20)
# plt.xlabel("Height")
# plt.ylabel("Frequency")
# plt.xlim(50, 90)
# plt.show()

# plt.scatter(all_data.height, all_data.income)
# plt.xlabel("Height")
# plt.ylabel("Income")
# plt.show()

# Many respondent did not provide their income level. 
# Therefore they're not useful for my analysis and will remove them.
# This will introduce bias: most probably people with higher income are more comfortable sharing
# their income level so they could be over-represented
print('n of all rows: ', len(all_data['income']))
all_data.reset_index(drop=True)
data_with_income = all_data.drop(all_data[all_data.income == -1].index)
print('n of rows with income: ', len(data_with_income['income']))
# We end up with roughly 1/6 of rows. I think this might introduce a bias?

# Transform categorical data

data_with_income['job'] = pd.Categorical(data_with_income['job'])
dfDummies_job = pd.get_dummies(data_with_income['job'], prefix = 'job')
data_with_income = pd.concat([data_with_income, dfDummies_job], axis=1)

# transform qualitative data into a quantitative scale

# For education will have
# 0: no education, 1: high school or below bachelor, 2: bachelor, 3: master (+ med/law), 4: phd or above
education_mapping = {
    'graduated from college/university': 2,
    'graduated from masters program': 3,
    'working on college/university': 1,
    'working on masters program': 2,
    'graduated from two-year college': 1,
    'graduated from high school': 1,
    'graduated from ph.d program': 4,
    'graduated from law school': 3,
    'working on two-year college': 1,
    'dropped out of college/university': 1,
    'working on ph.d program': 3,
    'college/university': 2,
    'graduated from space camp': 1,
    'dropped out of space camp': 0,
    'graduated from med school': 3,
    'working on space camp': 0,
    'working on law school': 2,
    'two-year college': 1,
    'working on med school': 2,
    'dropped out of two-year college': 1,
    'dropped out of masters program': 2,
    'masters program': 3,
    'dropped out of ph.d program': 3,
    'dropped out of high school': 0,
    'high school': 1,
    'working on high school': 0,
    'space camp': 0,
    'ph.d program': 4,
    'law school': 3,
    'dropped out of law school': 2,
    'dropped out of med school': 2,
    'med school': 3,
}
data_with_income['education_level'] = data_with_income.education.map(education_mapping)
data_with_income['education_level'] = data_with_income['education_level'].replace(np.nan, -1, regex=True)

# Normalize the quantitative data

data_with_income['height'] = data_with_income['height'].replace(np.nan, 0, regex=True)
data_with_income['income'] = data_with_income['income'].replace(np.nan, 0, regex=True)

feature_data = data_with_income[['height', 'income', 'education_level']]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

print('--- Regression income given height ---')

# print(data_with_income[['height', 'income']])
X = data_with_income[['height']]
Y = data_with_income[['income']]
# plt.scatter(X, Y)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, Y)

mlr = LinearRegression()
mlr.fit(x_train, y_train)

print(mlr.score(x_train, y_train))
print(mlr.coef_)
print(mlr.intercept_)

print('--- Regression mlr income given education level ---')

# print(data_with_income[['height', 'education_level']])
X = data_with_income[['education_level']]
Y = data_with_income[['income']]
# plt.scatter(X, Y)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, Y)

mlr = LinearRegression()

start = time.time()
mlr.fit(x_train, y_train)
end = time.time()
print('mlr fit time: ', end - start)

print('mlr score', mlr.score(x_test, y_test))

y_predicted_mlr = mlr.predict(x_test)
# print('mlr metrics: ', metrics.classification_report(y_test, y_predicted_mlr))

print('--- KN Regressor income given education level ---')

regressor = KNeighborsRegressor(n_neighbors=3, weights='distance')

start = time.time()
regressor.fit(x_train, y_train)
end = time.time()
print('kn regressor fit time: ', end - start)

print('knr score', regressor.score(x_test, y_test))

y_predicted_knr = regressor.predict(x_test)
# print('knr metrics: ', metrics.classification_report(y_test, y_predicted_knr))

# Guess job label given education level and income

# For job will combine the following:
# 0: other (+ rather not say), 1: student, 2: unemployed, 3: retired,
# 4: engineering (+ computer) ... the rest will get its own code
job_mapping = {
    'other': 0,
    'student': 1,
    'science / tech / engineering': 4,
    'computer / hardware / software': 4,
    'artistic / musical / writer': 5,
    'sales / marketing / biz dev': 6,
    'medicine / health': 7,
    'education / academia': 8,
    'executive / management': 9,
    'banking / financial / real estate': 10,
    'entertainment / media': 11,
    'law / legal services': 12,
    'hospitality / travel': 13,
    'construction / craftsmanship': 14,
    'clerical / administrative': 15,
    'political / government': 16,
    'rather not say': 0,
    'transportation': 17,
    'unemployed': 2,
    'retired': 3,
    'military': 18,
}
data_with_income['job_labels'] = data_with_income.job.map(job_mapping)
data_with_income['job_labels'] = data_with_income['job_labels'].replace(np.nan, 0, regex=True)

datapoints = data_with_income[['education_level', 'income']]
labels = data_with_income['job_labels']
# print(datapoints);
# plt.scatter(X, Y)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(datapoints, labels, random_state = 42)

print('--- KNeighborsClassifier ---')

knn = KNeighborsClassifier(n_neighbors=5)

start = time.time()
knn.fit(x_train, y_train)
end = time.time()
print('knn fit time: ', end - start)

y_predicted = knn.predict(x_test)
score = knn.score(x_test, y_test)
print('knn score: ', score)
print('knn metrics: ', metrics.classification_report(y_test, y_predicted))

print('--- SVC classifier ---')

classifier = SVC(kernel='rbf', gamma=0.1)

start = time.time()
classifier.fit(x_train, y_train)
end = time.time()
print('svc fit time: ', end - start)

y_predicted_svc = classifier.predict(x_test)
score_svc = classifier.score(x_test, y_test)
print('SVC score: ', score_svc)
print('SVC metrics: ', metrics.classification_report(y_test, y_predicted_svc))
