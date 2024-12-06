import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import geopy as geo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # Handliing Feature Scaling
from sklearn.impute import SimpleImputer #Handling missing values
from sklearn.preprocessing import OrdinalEncoder #Ordinal Encoding
##Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
class dataCleaning:
    def __init__(self) -> None:
         pass
    def data_Cleaning(self,df):
        try:
            data=df.drop(labels=['ID','Delivery_person_ID'], axis=1)
            
            df = data.loc[data.isnull().sum(1)>=3]
            data = pd.concat([data, df, df]).drop_duplicates(keep=False)
            df = data[data['City'].notna()]
            for i in df.index:
                if df['Time_Orderd'][i] >= '05:00' and df['Time_Orderd'][i] < '10:00':
                    df.loc[i,'day_quaters'] = 'morning'
                elif df['Time_Orderd'][i] >= '10:00' and df['Time_Orderd'][i] < '14:00':
                    df.loc[i,'day_quaters'] = 'late morning'
                elif df['Time_Orderd'][i] >= '14:00' and df['Time_Orderd'][i] < '19:00':
                    df.loc[i,'day_quaters'] = 'afternoon'
                elif df['Time_Orderd'][i] >= '19:00':
                    df.loc[i,'day_quaters'] = 'night'
            df=df.drop(labels=['Time_Orderd','Time_Order_picked','Order_Date'], axis=1)
            df['Festival'].fillna("No", inplace = True) # 98% of the column has No as a value 
            df['day_quaters'].fillna("night", inplace = True) # most of the people ordered at night
            df['Delivery_person_Ratings'].fillna(round(np.mean(df['Delivery_person_Ratings']),1), inplace = True)
            df['Delivery_person_Age'].fillna(round(np.mean(df['Delivery_person_Age'])), inplace = True)
            df['multiple_deliveries'].fillna(round(np.mean(df['multiple_deliveries'])), inplace=True)
            # print(df.head())
            df = df.dropna()
            return df
        except Exception as e:
            print("Exception Occured")
if __name__ == "__main__":
    data = pd.read_csv("https://raw.githubusercontent.com/AKISHPOTHURI/DataScience/main/Dataset/online_order.csv")
    dataclass = dataCleaning()
    cleaneddata = dataclass.data_Cleaning(data)