from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from scipy import stats
import pandas as pd
import numpy as np

# Select Top Brands

class Select_Top_Brands(BaseEstimator, TransformerMixin):
    def __init__(self,num_brands=15,variables = None):
        self.num_brands = num_brands
            
    def fit(self,X,y=None):
        self.most_pop = []
        brands = pd.DataFrame(X.groupby('item_brand')['title'].count())
        brands.columns = ['title']
        brands = brands.nlargest(self.num_brands,['title'])
        self.most_pop = brands.index
        return self
    
    def transform(self,X):
        
        X = X.copy()
        X = X[X['item_brand'].isin(self.most_pop)]
        return X
        
# Select a particular category

class Generate_Pick_Category(BaseEstimator, TransformerMixin):
    def __init__(self,category=None):
        self.category = category
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        X['category'] = X['item_type'].apply(lambda cat: str(cat).split("/")[0])
        if self.category==None:
            return X
        else:
            X = X[X['category']==self.category]
            return X
        
# Outlier Removal

class Outlier_Removal(BaseEstimator, TransformerMixin):
    def __init__(self,var_interest=None):
        self.var_interest = var_interest
    
    def fit(self,X,y=None):
        return self
        
    def transform(self,X):
        X = X.copy()
        X =X[(np.abs(stats.zscore(X[self.var_interest])) < 3)]
        return X

# Discrete Imputer

class Discrete_Imputer(BaseEstimator, TransformerMixin):
    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self,X,y=None):
        self.mode_dict = {}
        for var in self.variables:
            self.mode_dict[var] = X[var].mode()[0]
        return self
    
    def transform(self,X):
        X = X.copy()
        for var in self.variables:
            X[var].fillna(self.mode_dict[var],inplace=True)
        return X

# Categorical Imputer
        
class Continuos_Imputer(BaseEstimator, TransformerMixin):
    def __init__(self,variables = None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables
    
    def fit(self,X,y=None):
        self.mean_dict = {}
        for var in self.variables:
            self.mean_dict[var] = X[var].mean()
        return self
    
    def transform(self,X):
        X = X.copy()
        for var in self.variables:
            X[var].fillna(self.mean_dict[var],inplace=True)
        return X
    
# Log Transformer

class Log_Transformation(BaseEstimator, TransformerMixin):
    
    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables
            
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy() 
        for var in self.variables:
            X[var] = np.log1p(X[var])
        return X

# Feature Selector

class Top_Feature_Selection(BaseEstimator, TransformerMixin):
    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables
            
    def fit(self,X,y=None):
        self.best_features = []
        sel_ = SelectFromModel(Lasso(alpha=0.005, random_state=0)) 
        sel_.fit(X[self.variables],y)
        self.best_features = X[self.variables].columns[(sel_.get_support())]
        return self
    
    def transform(self,X):
        X = X.copy()
        self.best_features = self.best_features
        X = X[self.best_features]
        return X
    
# Feature Scaler
    
class Normalization(BaseEstimator, TransformerMixin):
    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables
            
    def fit(self,X,y=None):
        self.scaler_norm = MinMaxScaler()
        self.scaler_norm.fit(X[self.variables])
        return self
    
    def transform(self,X):
        X = X.copy()
        X[self.variables] = self.scaler_norm.transform(X[self.variables])
        return X
    
