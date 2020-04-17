import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.linear_model as lm
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

house_df = pd.read_csv(r'C:\Users\amin\Desktop\house_okay.csv', encoding='utf-8')
# house_df.rename(columns={'目标价格(元/m²)':'目标价格(元/平方米)'}, inplace=True)
#
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False

def summerize_data(df):
    for column in df.columns:
        print(column)
        if df.dtypes[column] == np.object:  # Categorical data
            print(df[column].value_counts())
        else:
            print(df[column].describe())
        print('\n')

summerize_data(house_df)

def price(row):
    i = row['Price_target']
    if i >45000:
        str = '高'
    elif 27000 < i <= 45000:
        str = '中等'
    else:
        str = '低'
    return str

house_df['Price'] = house_df.apply(lambda x:price(x), axis=1)
house_df.drop(['Price_target'], axis=1, inplace=True)

def number_encode_features(df):
    result = df.copy()

    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column].astype(str))
    return result, encoders

encoded_data, _ = number_encode_features(house_df)
print(encoded_data, _)

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(encoded_data.corr(), annot=False, square=True, ax=ax)
plt.show()

new_series = encoded_data["Price"]
# new_series = encoded_data["Price_target"]

X_train, X_test, y_train, y_test =\
ms.train_test_split(encoded_data[encoded_data.columns.drop(["Price"])],
new_series, train_size=0.8)
# X_train, X_test, y_train, y_test = \
#     ms.train_test_split(encoded_data[encoded_data.columns.drop(["Price_target"])],
#                         new_series, train_size=0.8)
scaler = preprocessing.StandardScaler()  # 进行标准化和归一化的类
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = scaler.transform(X_test)
print(X_train)

# logistic回归
cls = lm.LogisticRegression()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
print(accuracy_score(y_test, y_pred))

figsize = (6, 5)
coefs = pd.Series(cls.coef_[0], index=X_train.columns)
print(X_train.columns)
coefs = coefs.sort_values()
plt.subplot(1,1,1)
coefs.plot(kind="bar")
plt.xticks(rotation=30, fontsize=12)
plt.show()
print(coefs.sort_values(ascending = True))