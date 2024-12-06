from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pandas as pd
import numpy as np

from src.FoodDelivery.components.stage_1_data_cleaning import dataCleaning
from src.FoodDelivery.components.stage_2_data_Transformation import dataTransformation

class modelTraining:
    def __init__(self) -> None:
         pass
    # def evaluate_model(true, predicted):
    #     mae = mean_absolute_error(true, predicted)
    #     mse = mean_squared_error(true, predicted)
    #     rmse = np.sqrt(mean_squared_error(true, predicted))
    #     r2_square = r2_score(true, predicted)
    #     return mae, rmse, r2_square

    def model_training(self,X_train,y_train,X_test,y_test):
        regression = LinearRegression()
        regression.fit(X_train,y_train)
        models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
        }

        trained_model_list=[]
        model_list=[]
        r2_list=[]
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,y_train)

            #Make Predictions
            y_pred=model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2_square = r2_score(y_test, y_pred)
            # mae, rmse, r2_square=evaluate_model(y_test,y_pred)

            print(list(models.keys())[i])
            model_list.append(list(models.keys())[i])

            print('Model Training Performance')
            print("RMSE:",rmse)
            print("MAE:",mae)
            print("R2 score",r2_square*100)

            r2_list.append(r2_square)
            
            print('='*35)
            return model
    def modelPredict(self,new_input,model):
            output = model.predict(new_input)
            print(output)
            return output




if __name__ == "__main__":
    data = pd.read_csv("https://raw.githubusercontent.com/AKISHPOTHURI/DataScience/main/Dataset/online_order.csv")
    dataclass = dataCleaning()
    cleaneddata = dataclass.data_Cleaning(data)
    print("------data cleaning done---------------")
    datatransformation = dataTransformation()
    datatransformed = datatransformation.calc_distance(cleaneddata)
    print("------distance cal done---------------")
    X_train, X_test, y_train, y_test = datatransformation.train_test(datatransformed)
    print("------train_test_split done-----------")
    training = modelTraining()
    model = training.model_training(X_train,y_train,X_test,y_test)
    print("------model Training done-------------")
    new_input = [[36.0,3,30.327968,78.046106,30.397968,78.116106,1,1,2,1,1,3.0,2,1,1,10.280582]]
    modelOutput = training.modelPredict(new_input,model)
    print("------model Prediction done-----------")
    print("model output: ",modelOutput)

