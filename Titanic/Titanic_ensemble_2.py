#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sns.set(style='white', context='notebook', palette='deep')


print('================S T A R T================')
print('=========================================')
# Load data
##### Load train and Test set
# train = pd.read_csv('~/Desktop/Daeheon/ECU-Test/KaggleForMe/Titanic/train.csv')
# test = pd.read_csv('~/Desktop/Daeheon/ECU-Test/KaggleForMe/Titanic/train.csv')
# IDtest = test['PassengerId']
train = pd.read_csv('~/Desktop/Daeheon/conda/KaggleForMe/Titanic/train.csv')
test = pd.read_csv('~/Desktop/Daeheon/conda/KaggleForMe/Titanic/train.csv')
IDtest = test['PassengerId']

# Outlier detection 
# 이상치 감지 - 범위에서 많이 벗어난 값
def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
# Show the outliers rows
print(train.loc[Outliers_to_drop] )
# Drop outliers
train = train.drop(Outliers_to_drop, axis= 0).reset_index(drop= True)
print('=========================================')
# 2.3 Joining train and test set
# Join train and test datasets in order to obtain the same number of features during categorical conversion
train_len = len(train)
dataset = pd.concat(objs=[train, test], axis= 0).reset_index(drop=True)

# 2.4 Checkfor null and missing values
# Fill empty and NaNs values with NaN
dataset = dataset.fillna(np.nan)
# Check for Null values
print(dataset.isnull().sum())
print('=========================================')
# Infomations
print(train.info())
print(train.isnull().sum())
print(train.head())
print(train.dtypes)     #.info() 에도 내용이 있는데, 그 중 type만 꺼내온 것 같음
print(train.describe()) # 통계치 내주는 함수
print('=============FEATURE ANALYSIS=============')
# Numerical values
# Correlation Matrix between numerical values (SibSp, Age, Parch and Fare values) and Survived
g = sns.heatmap(train[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot=True, fmt='.2f', cmap='coolwarm')
# Fare & Survived 상관관계 있어 보임
# 다른 상관관계도 파악해보기
print('==============SibSp vs Survived======================')
# Explore SibSp feature vs Survived
g = sns.catplot(x='SibSp', y='Survived', data=train, kind='bar')
g.despine(left=True)
g = g.set_ylabels('Survival probability')
# A lot of SibSp have less chance to survive
print('==============Parch vs Survived======================')
g = sns.catplot(x='Parch', y='Survived', data=train, kind='bar')
g.despine(left=True)
g = g.set_ylabels('Survival probability')
# Small families have more chance to survive
print('==============Age vs Survived======================')
# Numberical feature 이므로 bar 형식보다는 histogram  형식이 나아 보임
# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.histplot, 'Age', kde=True, stat="density", linewidth=1)







#%%