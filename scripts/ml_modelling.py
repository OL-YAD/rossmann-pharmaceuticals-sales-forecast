import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from datetime import datetime


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



def build_model():
    # Manually tuned parameters due to resource
    rf = RandomForestRegressor(
        n_estimators=200, 
        max_depth=64, 
        criterion='squared_error',
        min_samples_split=10, 
        min_samples_leaf=2, 
        n_jobs=-1, 
        random_state=42)

      
    return rf


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    msle = mean_squared_log_error(y_test, y_pred)
    return mse, mae, rmse, r2, msle

def get_feature_importance(model, preprocessor):
    # Extract the names of numeric and one-hot encoded categorical features
    numeric_features = preprocessor.transformers_[0][2]
    categorical_transformer = preprocessor.transformers_[1][1]  # The pipeline that has OneHotEncoder
    categorical_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(preprocessor.transformers_[1][2])
    
    # Combine both numeric and categorical feature names
    feature_names = list(numeric_features) + list(categorical_features)
    
    # Get feature importance from the model
    feature_importance = model.feature_importances_
    
    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    return importance_df.sort_values('importance', ascending=False)



def calculate_confidence_interval(y_pred, confidence=0.95):
    n = len(y_pred)
    m = np.mean(y_pred)
    se = np.std(y_pred, ddof=1) / np.sqrt(n)
    h = se * np.abs(np.random.standard_t(df=n-1, size=1))
    return m - h, m + h


def serialize_model(model, path):
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")
    filename = f'model_{timestamp}.pkl'
    full_path=os.path.join(path,filename)
    joblib.dump(model, full_path)
    return filename



if __name__ == "__main__":
    # Load the data
    X_train, X_test, y_train = load_data()

    # Create the preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Split the training data further for evaluation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Model building and training
    model = build_model()
    trained_model = train_model(model, X_train, y_train)
    
    # Evaluate the model
    mse, mae, rmse, r2, msle = evaluate_model(trained_model, X_val, y_val)
    print(f"MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R2: {r2}, MSLE: {msle}")
    
    # Get feature importance with actual column names
    feature_importance = get_feature_importance(trained_model, preprocessor)
    print("Top 10 important features:")
    print(feature_importance.head(10))
    
    # Predictions and confidence intervals
    y_pred = trained_model.predict(X_val)
    lower_ci, upper_ci = calculate_confidence_interval(y_pred)
    print(f"95% Confidence Interval: ({lower_ci}, {upper_ci})")
    
    # Serialize the model
    model_filename = serialize_model(trained_model)
    print(f"Model serialized and saved as: {model_filename}")
