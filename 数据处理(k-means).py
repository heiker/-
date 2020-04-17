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

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def summerize_data(df):
    for column in df.columns:
        print(column)
        if df.dtypes[column] == np.object:  # Categorical data
            print(df[column].value_counts())
        else:
            print(df[column].describe())
        print('\n')

summerize_data(house_df)

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

plt.rc('font', family='STXihei', size=10)
plt.scatter(encoded_data['Price_target'],encoded_data['develop_loca'],50,color='blue',marker='+',linewidth=2,alpha=0.8)
plt.xlabel('Price_target')
plt.ylabel('develop_loca')
plt.xlim(0,40000)
plt.grid(color='#95a5a6',linestyle='--', linewidth=1,axis='both',alpha=0.4)
plt.show()


li = encoded_data.columns.to_list()
loan = np.array(encoded_data[li])
loan = np.array(encoded_data[['Price_target', 'layout ']])
#设置类别为3
clf=KMeans(n_clusters=3)
#将数据代入到聚类模型中
clf=clf.fit(loan)
print(clf.cluster_centers_)

house_df['label']=clf.labels_
loan_data0=encoded_data.loc[house_df["label"] == 0]
loan_data1=encoded_data.loc[house_df["label"] == 1]
loan_data2=encoded_data.loc[house_df["label"] == 2]
loan_data3=encoded_data.loc[house_df["label"] == 3]

plt.rc('font', family='STXihei', size=10)
plt.scatter(loan_data0['Price_target'],loan_data0['layout '],50,color='red',marker='+',linewidth=2,alpha=0.8)
plt.scatter(loan_data1['Price_target'],loan_data1['layout '],50,color='blue',marker='*',linewidth=2,alpha=0.8)
plt.scatter(loan_data2['Price_target'],loan_data2['layout '],50,color='yellow',marker='x',linewidth=2,alpha=0.8)
plt.xlabel('Price_target')
plt.ylabel('layout ')
plt.xlim(0,40000)
plt.grid(color='#95a5a6',linestyle='--', linewidth=1,axis='both',alpha=0.4)
plt.show()
