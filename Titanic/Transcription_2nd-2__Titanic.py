# %%
import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
# matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고,
# 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.
sns.set(font_scale=2.5) 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# EDA 에서 작업했던 내용 중 Feature Engineering에 써야할 자료 import 하기
df_train = pd.read_csv('~/Desktop/Daeheon/conda/KaggleForMe/Titanic/train.csv')
df_test = pd.read_csv('~/Desktop/Daeheon/conda/KaggleForMe/Titanic/test.csv')
# SibSp + Parch 를 이용하여 FamilySize 만들기
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
# Test set 에서 Fare 의 Null값을 평균으로 넣어주기
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()
# Fare는 너무 비대칭이 심하여 log 적용하여 비대칭 줄이기
df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

print('--------------S T A R T--------------')
# Data 의 Null 값 채우기
# train set의 NaN를 변경한 것과 동일하게 Test Set도 적용해주어야함
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Age's NaN value
# Mr, Miss, Mrs 와 같은 title이 존재함
# pandas's  str()(data->string으로 변환) & extract()(정규표현식 적용) 사용하여 추출
# str.extract('([정규표현식])') <- 바로 이 위치에 추출하고 싶은 문자열의 정규표현식을 대입하면 된다
# 해당 정규표현식은 .앞에 있는 문자열을 추출하는 것 같아 보임
df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')
df_test['Initial'] = df_test.Name.str.extract('([A-Za-z]+)\.')
#Checking the Initials with the Sex
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')

# 남자 & 여자의 Initial을 구분하기< replace() 사용>
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                            ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                           ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
df_train.groupby('Initial')['Survived'].mean().plot.bar()

# Age의 Null data 넣는 방식
# 1. statistics 를 활용하는 방법도 있고,
# 2. null data 가 없는 데이터를 기반으로 새로운 머신러닝 알고리즘을 만들어 예측해서 채워넣는 방식

# Statistics를 활용할 것이며, 이는 Train data만 가지고 사용해야함
# Test set은 언제나 Unseen 상태로 두어야하니께!

print(df_train.groupby('Initial').mean())
# 평균값을 이용하여 Null값 채우기
# boolean array를 사용하여 indexing 하기
# isnull() 이면서 Initial이 Mr인 조건 을 만족 시 탑승객의 Age를 33으로 변경
# loc + boolean+ column
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial=='Mr'), 'Age'] = 33
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial=='Mrs'),'Age'] = 36
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial=='Master'),'Age'] = 5
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial=='Miss'),'Age'] = 22
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial=='Other'),'Age'] = 46

df_test.loc[(df_test.Age.isnull()) & (df_test.Initial=='Mr'),'Age'] = 33
df_test.loc[(df_test.Age.isnull()) & (df_test.Initial=='Mrs'),'Age'] = 36
df_test.loc[(df_test.Age.isnull()) & (df_test.Initial=='Master'),'Age'] = 5
df_test.loc[(df_test.Age.isnull()) & (df_test.Initial=='Miss'),'Age'] = 22
df_test.loc[(df_test.Age.isnull()) & (df_test.Initial=='Other'),'Age'] = 46

# 다른 방식으로 Age를 채우는 방법
'''
4. Filling missing Values
4.1 Age
As we see, Age column contains 256 missing values in the whole dataset.

Since there is subpopulations that have more chance to survive (children for example), it is preferable to keep the age feature and to impute the missing values.

To adress this problem, i looked at the most correlated features with Age (Sex, Parch , Pclass and SibSP).

---------------------------C O D E ---------------------------
# Explore Age vs Sex, Parch , Pclass and SibSP
g = sns.factorplot(y="Age",x="Sex",data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="box")
---------------------------------------------------------------



Age distribution seems to be the same in Male and Female subpopulations, so Sex is not informative to predict Age.

However, 1rst class passengers are older than 2nd class passengers who are also older than 3rd class passengers.

Moreover, the more a passenger has parents/children the older he is and the more a passenger has siblings/spouses the younger he is.
---------------------------C O D E ---------------------------
# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
---------------------------------------------------------------
The correlation map confirms the factorplots observations except for Parch. Age is not correlated with Sex, but is negatively correlated with Pclass, Parch and SibSp.

In the plot of Age in function of Parch, Age is growing with the number of parents / children. But the general correlation is negative.

So, i decided to use SibSP, Parch and Pclass in order to impute the missing ages.

The strategy is to fill Age with the median age of similar rows according to Pclass, Parch and SibSp.
---------------------------C O D E ---------------------------
# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med
---------------------------------------------------------------
/opt/conda/lib/python3.6/site-packages/pandas/core/indexing.py:179: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  self._setitem_with_indexer(indexer, value)
  
---------------------------C O D E ---------------------------
g = sns.factorplot(x="Survived", y = "Age",data = train, kind="box")
g = sns.factorplot(x="Survived", y = "Age",data = train, kind="violin")
--------------------------------------------------------------

No difference between median value of age in survived and not survived subpopulation.

But in the violin plot of survived passengers, we still notice that very young passengers have higher survival rate.
'''
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Embarked의 Null 값 채우기
print('Embarked has ', sum(df_train['Embarked'].isnull()), ' Null values')
# S에 가장 많은 탑승객이 있으므로 2개는 S로 채움
df_train['Embarked'].fillna('S', inplace=True)
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Age는 현재 Continuous Feature 이지만, Group화 시켜 Category화 예정
# 자칫 Information Loss가 생길 수도 있지만, 다양한 방법을 위해 진행

# 1. loc를 사용한 방법
df_train['Age_cat'] = 0
df_train.loc[df_train['Age'] < 10, 'Age_cat'] = 0
df_train.loc[(10 <= df_train['Age']) & (df_train['Age'] < 20), 'Age_cat'] = 1
df_train.loc[(20 <= df_train['Age']) & (df_train['Age'] < 30), 'Age_cat'] = 2
df_train.loc[(30 <= df_train['Age']) & (df_train['Age'] < 40), 'Age_cat'] = 3
df_train.loc[(40 <= df_train['Age']) & (df_train['Age'] < 50), 'Age_cat'] = 4
df_train.loc[(50 <= df_train['Age']) & (df_train['Age'] < 60), 'Age_cat'] = 5
df_train.loc[(60 <= df_train['Age']) & (df_train['Age'] < 70), 'Age_cat'] = 6
df_train.loc[70 <= df_train['Age'], 'Age_cat'] = 7

df_test['Age_cat'] = 0
df_test.loc[df_test['Age'] < 10, 'Age_cat'] = 0
df_test.loc[(10 <= df_test['Age']) & (df_test['Age'] < 20), 'Age_cat'] = 1
df_test.loc[(20 <= df_test['Age']) & (df_test['Age'] < 30), 'Age_cat'] = 2
df_test.loc[(30 <= df_test['Age']) & (df_test['Age'] < 40), 'Age_cat'] = 3
df_test.loc[(40 <= df_test['Age']) & (df_test['Age'] < 50), 'Age_cat'] = 4
df_test.loc[(50 <= df_test['Age']) & (df_test['Age'] < 60), 'Age_cat'] = 5
df_test.loc[(60 <= df_test['Age']) & (df_test['Age'] < 70), 'Age_cat'] = 6
df_test.loc[70 <= df_test['Age'], 'Age_cat'] = 7
 
 # 2. apply() 사용
def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7   
df_train['Age_cat_2'] = df_train['Age'].apply(category_age)

# 두 결과는 같은 값을 내야함
print('1번 방법, 2번 방법 둘다 같은 결과를 내면 True 줘야함 -> ', 
      (df_train['Age_cat'] == df_train['Age_cat_2']).all())

df_train.drop(['Age', 'Age_cat_2'], axis=1, inplace=True)
df_test.drop(['Age'], axis=1, inplace=True)
# Age를 유추하였으니, Initial도 카테고리화
df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})


# Embarked 확인
print(df_train['Embarked'].unique())
print(df_train['Embarked'].value_counts())
# C, Q, S 를 각각 숫자로 변환해줌
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
# Null값 있나 다시한번 확인
print(df_train['Embarked'].isnull().any())
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Sex도 0, 1로 변환
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
print('----------------------------------------------------------------------------------------------------------------------------------------')
# 상관관계 확인
# 두 변수간 Pearson correlation을 구하기
# (-1, 1) 사이의 값으로 나타나며, 음의 상관관계, 양의 상관관계, 0 = 관계없음 을 의미함
heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']] 

colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})

del heatmap_data
print('----------------------------------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Data Preprocessing (데이터 전처리!!!!)
# One-hot encoding on Initial and Embarked
# Initial 과 Embarked는 수치화 시켰으니 그냥 넣어도 되지만, 
# 모델의 성능을 높이기 위해 One-hot Encoding 적용
# 분류문제는 n개의 카테고리 인코딩 / 회귀문제는 n-1개 인코딩

# 1. pandas의 get_dummies 사용
df_train = pd.get_dummies(df_train, columns= ['Initial'], prefix= ['Initial'])
df_test = pd.get_dummies(df_test, columns= ['Initial'], prefix= ['Initial'])
print(df_train.head())
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')
df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')

# 2. sklearn의 Labelencoder + OneHotencoder 사용 가능
# Category가 100개가 넘어가는 경우 다른 방법 사용
# Labelencoder, Orinal-encoding, Helmert-encoding, Binary-encoding, Frequency-encoding, Mean-encoding, Weight of Evidence(WoE), Probability Ratio of encoding, Hashing, Backward difference encoding, Leave one out encoding, James-stein encoding, M-estimator encoding
# https://techblog-history-younghunjo1.tistory.com/99
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Drop columns
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
print('----------------------------------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Building machine Learning Model and prediction using the trained model
#importing all the required ML packages
from sklearn.ensemble import RandomForestClassifier # 유명한 randomforestclassfier 입니다. 
from sklearn import metrics # 모델의 평가를 위해서 씁니다
from sklearn.model_selection import train_test_split # traning set을 쉽게 나눠주는 함수입니다.

# Target lable(Survived) 를 분리
X_train = df_train.drop('Survived', axis=1).values
target_label = df_train['Survived'].values()
X_test = df_test.values()

# Train, validation,으로 분리
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Model generation and prediction
model = RandomForestClassifier()
model.fit(X_tr, y_tr)
prediction = model.predict(X_vld)
print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Feature importance
from pandas import Series

feature_importance = model.feature_importances_
Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8, 8))
Series_feat_imp.sort_values(ascending=True).plot.barh()
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Prediction on Test set
submission = pd.read_csv('~/Desktop/Daeheon/conda/KaggleForMe/Titanic/gender_submission.csv')
print(submission.head())
print('\n\n\n\n\n\n\n\n\n\n\n\n')
print('End')
print('----------------------------------------------------------------------------------------------------------------------------------------')
# %%