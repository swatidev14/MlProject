from data_ingestion import data
print(data.head())
#finding the duplicate 
print(data.isna().sum())
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

X = data.drop(columns=['math_score'],axis=1)
y=data['math_score']

num_features = [feature for feature in X.columns if X[feature].dtype!='O']
cat_features = [feature for feature in X.columns if X[feature].dtype=='O']

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

## we are making a pipeline here as in, this should first do the one hot encoding and the standardisation as one hot encoding will only happen for categoical features 
numeric_transformer = StandardScaler()
one_hot_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", one_hot_transformer, cat_features),
        ("StandardScaler", numeric_transformer, num_features),
    ]
)

X = preprocessor.fit_transform(X)

# Seperate the dataset into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

def evaluate_model(true, predicted):
    mse = mean_squared_error(true,predicted)
    mae= mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true,predicted))
    r2_square = r2_score(true, predicted)
    return mse, mae, rmse, r2_square


models = {
    "Linear Regression" : LinearRegression(),
    "Lasso" : Lasso(),
    "Ridge" : Ridge(),
    "K-Neighbors Regressor" : KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest" : RandomForestRegressor(),
    "XGBRegressor" : XGBRegressor(),
    "CatBoostRegressor ": CatBoostRegressor(verbose=False),
    "AdaBoostRegressor": AdaBoostRegressor()
}

model_list = []
r2_list = []

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train,y_train) #train_model

    #make prediction
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    #evaluate test and train dataset

    model_train_mse, model_train_mae, model_train_rmse, model_train_r2_score = evaluate_model(y_train,y_train_pred)
    model_test_mse, model_test_mae, model_test_rmse, model_test_r2_score = evaluate_model(y_test,y_test_pred)

    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print('Model performance for training set')
    print(" - Root mean square error : {: .4f}".format(model_train_rmse))
    print(" - Root mean absolure error : {: .4f}".format(model_train_mae))
    print(" - R2 Score : {: .4f}".format(model_train_mae))

    print('Model performance for testing set')
    print(" - Root mean square error : {: .4f}".format(model_test_rmse))
    print(" - Root mean absolure error : {: .4f}".format(model_test_mae))
    print(" - R2 Score : {: .4f}".format(model_test_mae))

    r2_list.append(model_test_r2_score)


print(pd.DataFrame(list(zip(model_list,r2_list)),columns=['Model Name', 'R2 Score']).sort_values(by='R2 Score',ascending=False))

print("\n \n Accuracy of the model ------------------- ")
# Accuracy of models 

print(pd.DataFrame(list(zip(model_list,[r2*100 for r2 in r2_list])),columns=['Model Name', 'Accuracy']).sort_values(by='Accuracy',ascending=False))

