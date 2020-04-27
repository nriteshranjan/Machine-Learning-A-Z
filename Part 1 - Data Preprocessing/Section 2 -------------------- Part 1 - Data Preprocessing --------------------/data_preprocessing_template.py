# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\rrite\Downloads\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#Taking care of missing data
# =============================================================================
# from sklearn.impute import SimpleImputer as Imputer
# imputer = Imputer(missing_values = np.nan, strategy = 'mean')
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])
# =============================================================================

#Encoding Categorial Data
# =============================================================================
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:,0] = labelencoder_X.fit_transform(X[:,0])
# from sklearn.compose import ColumnTransformer
# ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],remainder='passthrough')
# X = np.array(ct.fit_transform(X), dtype=np.float)
# labelencoder_Y = LabelEncoder()
# Y = labelencoder_Y.fit_transform(Y)
# =============================================================================

#Splitting the data set into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)
sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)
"""
