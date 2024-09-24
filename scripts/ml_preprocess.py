import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def extract_date_features(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['DayOfMonth'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Season'] = pd.cut(df['Month'], 
                          bins=[0, 3, 6, 9, 12], 
                          labels=['Winter', 'Spring', 'Summer', 'Fall'],
                          include_lowest=True)
    return df


# def calculate_holiday_distances(df):
#     # This is a placeholder. You'll need to implement the actual logic
#     # based on your holiday data and requirements
#     df['DaysToNextHoliday'] = 0
#     df['DaysFromPrevHoliday'] = 0
#     return df


def create_preprocessing_pipeline():
    numeric_features = ['Store', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Year', 'Month', 
                        'CompetitionDistance', 'CompetitionOpenSinceMonth', 
                        'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear',
                        ]
    categorical_features = ['StoreType', 'Assortment', 'StateHoliday', 'SchoolHoliday', 
                            'Season', 'Promo', 'Promo2']

    numeric_transformer = Pipeline(steps=[

        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor