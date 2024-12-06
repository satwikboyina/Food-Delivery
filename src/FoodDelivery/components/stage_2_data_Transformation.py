import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Handliing Feature Scaling
from sklearn.impute import SimpleImputer #Handling missing values
from sklearn.preprocessing import OrdinalEncoder #Ordinal Encoding
##Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from src.FoodDelivery.components.stage_1_data_cleaning import dataCleaning

class dataTransformation:
    def __init__(self) -> None:
         pass
    
    def calc_distance(self,df):    
          # Set the earth's radius (in kilometers)
          df = df.dropna()
          R = 6371
          index = []
          # Convert degrees to radians
          def deg_to_rad(degrees):
               return degrees * (np.pi/180)

          # Function to calculate the distance between two points using the haversine formula
          def distcalculate(lat1, lon1, lat2, lon2):
               d_lat = deg_to_rad(lat2-lat1)
               d_lon = deg_to_rad(lon2-lon1)
               a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
               c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
               return R * c
          
          # Calculate the distance between each pair of points
          # data1['distance'] = np.nan

          for i in range(len(df)):
               if i in df.index:
                    df.loc[i, 'distance'] = distcalculate(df.loc[i, 'Restaurant_latitude'], 
                                                  df.loc[i, 'Restaurant_longitude'], 
                                                  df.loc[i, 'Delivery_location_latitude'], 
                                                  df.loc[i, 'Delivery_location_longitude'])
               else:
                    index.append(i)   
          return df

    def train_test(self,df):
          X = df.drop(labels=['Time_taken (min)'], axis=1)
          Y = df[['Time_taken (min)']]
          categorical_cols = X.select_dtypes(include='object').columns
          numerical_cols = X.select_dtypes(exclude='object').columns
          city_map = ['Metropolitian','Urban','Semi-Urban']
          Festival_map = ['Yes','No']
          Road_traffic_density_map = ['Jam','High','Medium','Low']
          Type_of_order_map = ['Snack','Meal','Drinks','Buffet']
          Type_of_vehicle_map = ['motorcycle','scooter','electric_scooter']
          Weather_conditions_map = ["Sunny","Stormy","Sandstorms","Windy","Fog","Cloudy"]
          day_quaters_map = ['night','afternoon','morning','late morning']
          #Nuerical Pipeline
          num_pipeline = Pipeline(
               steps=[
               ('imputer', SimpleImputer(strategy='mean')),
               ('scaler', StandardScaler())
               ]
          )

          #Categorical Pipeline
          cat_pipeline = Pipeline(
               steps=[
               ('imputer', SimpleImputer(strategy='most_frequent')),
               ('OrdinalEncoder', OrdinalEncoder(categories=[Weather_conditions_map,Road_traffic_density_map,Type_of_order_map,Type_of_vehicle_map,Festival_map,
                                                            city_map,day_quaters_map])),
               ('scaler', StandardScaler())
               ]
          )

          preprocessor = ColumnTransformer([
          ('num_pipeline',num_pipeline,numerical_cols),
          ('cat_pipeline',cat_pipeline,categorical_cols)
          ])

          X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.30,random_state=42)
          X_train= pd.DataFrame(preprocessor.fit_transform(X_train),columns=preprocessor.get_feature_names_out())
          X_test = pd.DataFrame(preprocessor.transform(X_test),columns=preprocessor.get_feature_names_out())
          return X_train, X_test, y_train, y_test
    

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
