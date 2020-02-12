from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
import modules as mod

# List of continuous variables
CONT_VAR = ['item_condition','shipping_category','item_origin','approved_poster']

# List of discrete variables
DISC_VAR = ['item_characteristic_n','item_characteristic_p','post_stats'] 

# List of total variables
TOT_VAR = CONT_VAR + DISC_VAR

# Initiating the regressor
lasso = Lasso()

# Rendering an end to end deployment pipeline

pipe = Pipeline([('Discrete Imputer',mod.Discrete_Imputer(variables = DISC_VAR)),
                 ('Continuos Imputer',mod.Continuos_Imputer(variables=CONT_VAR)),
                 ('Log Transformer',mod.Log_Transformation(variables=CONT_VAR)), 
                 ('Feature Selector',mod.Top_Feature_Selection(variables = TOT_VAR)),
                 ('Feature Scaler', mod.Normalization(variables = CONT_VAR)),
                 ('Regressor',lasso)])
