# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set(font_scale=2.5)

import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline
# 위 구문의 경우 jupyter notebook에서 사용되는 구문이며, matplotlib의 이미지가 브라우져 내부에 자동으로 인라인되게 하는 명령어 이다.
# vscode - python코드로 작동시 #%% 라는 구문을 이용하여 그래프 확인 가능


# 데이터셋 확인하기
df_train = pd.read_csv('~/Desktop/Daeheon/conda/KaggleForMe/Titanic/train.csv')
df_test = pd.read_csv('~/Desktop/Daeheon/conda/KaggleForMe/Titanic/test.csv')

# pandas's describe() method
# feature가 가진 통계치들을 반환해준다. (count, mean, std, min, max, 25%, 50%, 75%) //단, NaN 값은 제외된다!
# 따라서 다른 열의 count가 다른것을 확인해보니, Null값이 존재하는 것으로 판단됨.
print(df_train.describe())
print(df_test.describe())

print('----------------------------------------------------------------------------------------------------------------------------------------')
# Null data check
print('df_train')
for col in df_train.columns :
    msg = 'column: {:>10}\t Percent of NaN value :{:.2f}%'.format(col, (df_train[col].isnull().sum() / df_train[col].shape[0]) * 100 )
    print(msg)
print('df_test')
for col in df_test.columns :
    msg = 'column: {:>10}\t Percent of NaN value :{:.2f}%'.format(col, (df_test[col].isnull().sum() / df_test[col].shape[0]) * 100 )
    print(msg)
print('----------------------------------------------------------------------------------------------------------------------------------------')
# If use msno, you can see the null data to easy
# msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
# msno.bar(df=df_train.iloc[:,:], figsize=(8,8), color=(0.8, 0.5, 0.2))

print('----------------------------------------------------------------------------------------------------------------------------------------')

# Target Label 확인
# what kind of distribution it has (distribution = 분포)
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot(x='Survived', data=df_train,ax=ax[1])
ax[1].set_title('Count plot - Survived')

plt.show()
print('----------------------------------------------------------------------------------------------------------------------------------------')


# EDA
# 시각화 Library는 소스코드를 정리해두어, 특정 목적에 맞게 참고하면 편리하다고 함

# Pclass - 티켓의 클래스(1등석, 2등석, 3등석)
# groupby 라는 함수를 이용할 것이며, 주어진 데이터를 그룹별로 묶은 후 데이터를 보기 위한 함수
print(df_train[['Pclass','Survived']].groupby(['Pclass'], as_index = True).count())
print(df_train[['Pclass','Survived']].groupby(['Pclass'], as_index = True).sum())
# crosstab 함수는 위 함수보다 보기 편함
# pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')

# Pclass의 생존율 평균
print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar())
# 위 항목으로 보기 위해서는 가장 상단의 Run cell을 통해 실시

# seaborn 함수를 통한 counplot으로 xmrwjd label에 따른 개수 확인 용이
y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot(x='Pclass', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
plt.show()
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Sex - 성별
# 성별에 따른 생존률을 확인 할 것
# 위 Pclass의 label확인과 비슷할 것으로 예상되며, 다시한번 코드를 적으며 비교를 해보겠습니다.
f, ax = plt.subplots(1,2, figsize=(18, 8))
df_train[['Sex','Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot(x='Sex', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()

# Sex의 생존률 평균
print(df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# crosstab & groupby로도 Pclass처럼 확인 가능

# Pclass & Sex 2가지의 조건에 관하여 생존이 어떻게 변화되는지 확인하기
# factorplot 이 catplot으로 변경되었다고 하며, catplot 쓰면 모양 이상해짐...
# sns.catplot(x='Sex', y='Survived', col='Pclass', data=df_train, size=9, aspect=1)
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Age 
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))
print('제일 나이 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))
print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))

# 생존에 따른 Age의 Histogram
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'Survived == 0'])
plt.show()

# Age를 Pclass별로 나누어서 Histogram 그려보기
plt.figure(figsize=(8, 6))
df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])
plt.show()

# 나이대가 변하면 생존률이 어떻게 변하는지 확인하기
cummulate_survival_ratio=[]
for i in range(1,80) :
    cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age']< i]['Survived']))

plt.figure(figsize=(7,7))
plt.plot(cummulate_survival_ratio)
plt.title('Survival rate change depending on range of Age', y=1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~x)')
plt.show
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Pclass, Sex, Age 3가지 모두를 함께 통합해보기
# violinplot 사용!
# x = Pcalss, Sex, y = distribution(Age) 

f, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=df_train, scale='count', split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

sns.violinplot(x='Sex', y='Age', hue='Survived', data =df_train, scale='count', split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Embarked - 탑승한 항구
f, ax = plt.subplots(1, 1, figsize=(7, 7))
print(df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax))

# 탑승 항구 별로 성별, 생존, Pclass로 나누어서 확인해보기
f,ax=plt.subplots(2, 2, figsize=(20,15))
sns.countplot(x='Embarked', data=df_train, ax=ax[0,0])
ax[0,0].set_title('(1) No. Of Passengers Boarded')
sns.countplot(x='Embarked', hue='Sex', data=df_train, ax=ax[0,1])
ax[0,1].set_title('(2) Male-Female Split for Embarked')
sns.countplot(x='Embarked', hue='Survived', data=df_train, ax=ax[1,0])
ax[1,0].set_title('(3) Embarked vs Survived')
sns.countplot(x='Embarked', hue='Pclass', data=df_train, ax=ax[1,1])
ax[1,1].set_title('(4) Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
print('----------------------------------------------------------------------------------------------------------------------------------------')
# SibSp, Parch(형제자매, 부모자녀)
# SibSp + Parch = Family
# 합쳐서 한번에 분석해보기
# 자신을 포함해야하므로 1을 더하기
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['FamilySize'] = df_test['SibSp'] | df_test['Parch'] + 1
print("Maximum size of Family: ", df_train['FamilySize'].max())
print("Minimum size of Family: ", df_train['FamilySize'].min())


# Family와 생존의 관계를 살펴보기
f,ax=plt.subplots(1, 3, figsize=(40,10))
sns.countplot(x='FamilySize', data=df_train, ax=ax[0])
ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)

sns.countplot(x='FamilySize', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Fare - 탑승 요금
# Age와 같이 Contious Feature 이며 Histogram으로 살펴보기 편함
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.kdeplot(df_train[df_train['Survived'] == 1]['Fare'], ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Fare'], ax=ax)
plt.legend(['Survived == 1', 'Survived == 0'])
plt.show()
# distplot을 이용한 histogram
# 비대칭도 확인
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')

# testset 에 있는 nan value 를 평균값으로 치환합니다.
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()
# 비대칭이 너무 심하기에 log를 취하기
# map or apply 함수 이용
df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')
# g는 vscode python에서 가장 마지막에 스스로 출력됨,,,
# 해당을 코드 중간에 출력되게 하는 방법은 확인이 필요해보임

# 사실 log를 취하는 것은 EDA가 아닌, Feature Engineering이지만, 해봄

print('----------------------------------------------------------------------------------------------------------------------------------------')
# Cabin - 객실 번호
# 해당 Feature는 NaN이 80% 이상이므로, 유의미한 정보를 얻기 쉽지않음
# 따라서 제외
print('----------------------------------------------------------------------------------------------------------------------------------------')
# Ticket - 티켓 번호
# NaN은 없지만, string Data이므로 작업을 해주어야지 실제 모델에 사용 가능해짐
print(df_train['Ticket'].value_counts())

# 위 다양한 ticket number를 작업해주는 것 -> feature engineering이며
# 이것을 해결해 나가는 것이 Kaggle 이라고 함
print('\n\n\n\n\n\n\n\n\n\n\n\n')
print('End')


# %%
