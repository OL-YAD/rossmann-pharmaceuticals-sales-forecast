import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import holidays

def compare_promo_distribution(train_data, test_data):
    # Get the value counts and proportions
    train_promo_counts = train_data['Promo'].value_counts()
    test_promo_counts = test_data['Promo'].value_counts()
    
    # Calculate the percentage for each category
    train_promo_percentage = (train_promo_counts / len(train_data)) * 100
    test_promo_percentage = (test_promo_counts / len(test_data)) * 100
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(2)
    
    # Plot bars
    plt.bar(index, train_promo_percentage, bar_width, label='Train')
    plt.bar(index + bar_width, test_promo_percentage, bar_width, label='Test')
    
    # Add the actual percentages on top of each bar
    for i, v in enumerate(train_promo_percentage):
        plt.text(i - bar_width/2, v + 0.5, f'{v:.2f}%', ha='center')
        
    for i, v in enumerate(test_promo_percentage):
        plt.text(i + bar_width/2, v + 0.5, f'{v:.2f}%', ha='center')

    # Labels and title
    plt.xlabel('Promo')
    plt.ylabel('Percentage')
    plt.title('Distribution of Promotions in Train and Test Sets')
    plt.xticks(index + bar_width/2, ['No Promo', 'Promo'])
    plt.legend()
    plt.show()


def analyze_holiday_sales(df):
    # Ensure 'Date' is in datetime format
    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Analyze StateHoliday
    plt.figure(figsize=(12, 6))
    sns.barplot(x='StateHoliday', y='Sales', data=df)
    plt.title('Sales Distribution on State Holidays vs Non-Holidays')
    plt.xlabel('State Holiday (0: No, 1: Yes)')
    plt.ylabel('Sales')
    plt.show()

    # Analyze SchoolHoliday
    plt.figure(figsize=(12, 6))
    sns.barplot(x='SchoolHoliday', y='Sales', data=df)
    plt.title('Sales Distribution on School Holidays vs Non-Holidays')
    plt.xlabel('School Holiday (0: No, 1: Yes)')
    plt.ylabel('Sales')
    plt.show()

# sales distribution before, during and after holidays
def holiday_sales(df_train):
    df_train['DayType'] = 'Regular'
    df_train.loc[df_train['StateHoliday'] != '0', 'DayType'] = 'Holiday'
    df_train['DayType'] = pd.Categorical(df_train['DayType'], categories=['Before Holiday', 'Holiday', 'After Holiday', 'Regular'], ordered=True)

    # Mark days before and after holidays
    holiday_dates = df_train[df_train['DayType'] == 'Holiday']['Date'].unique()
    for date in holiday_dates:
        before_holiday = date - pd.Timedelta(days=1)
        after_holiday = date + pd.Timedelta(days=1)
        df_train.loc[df_train['Date'] == before_holiday, 'DayType'] = 'Before Holiday'
        df_train.loc[df_train['Date'] == after_holiday, 'DayType'] = 'After Holiday'

    plt.figure(figsize=(12, 6))
    sns.barplot(x='DayType', y='Sales', data=df_train)
    plt.title('Sales Distribution Before, During, and After State Holidays')
    plt.show()


# Seasonal Purchase Behaviors
def analyze_seasonal_behavior(df_train):
    df_train['Month'] = df_train['Date'].dt.month
    
    # Create a holiday list for the country  United States
    us_holidays = holidays.US(years=df_train['Date'].dt.year.unique())
    
    # Map each date to a holiday name or None
    df_train['Holiday'] = df_train['Date'].apply(lambda x: us_holidays.get(x))
    
    # Mark whether it's a holiday or not
    df_train['IsHoliday'] = df_train['Holiday'].notnull()
    
    # Calculate average sales during holidays
    holiday_sales = df_train[df_train['IsHoliday']].groupby('Holiday')['Sales'].mean().reset_index()
    
    # Plotting Monthly Sales
    plt.figure(figsize=(12, 6))
    monthly_sales = df_train.groupby('Month')['Sales'].mean().reset_index()
    sns.lineplot(x='Month', y='Sales', data=monthly_sales, marker='o')
    plt.title('Average Monthly Sales')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.xticks(range(1, 13))
    plt.show()
    
    # Plotting Holiday Sales
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Holiday', y='Sales', data=holiday_sales)
    plt.xticks(rotation=90)
    plt.title('Average Sales During Holidays')
    plt.xlabel('Holiday')
    plt.ylabel('Average Sales')
    plt.show()


# sales vs customer correlation 
def analyze_sales_customers_correlation(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Customers', y='Sales')
    plt.title('Sales vs Number of Customers')
    plt.show()

    correlation = df['Sales'].corr(df['Customers'])
    print(f"Correlation between Sales and Customers: {correlation}")

    # Create a correlation matrix
    corr_matrix = df[['Sales', 'Customers']].corr()

    # Heatmap for the correlation matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap: Sales vs Customers')
    plt.show()

def analyze_promo_effect(df):
    promo_effect = df.groupby('Promo')[['Sales', 'Customers']].mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    promo_effect['Sales'].plot(kind='bar', ax=ax1)
    ax1.set_title('Average Sales with/without Promo')
    ax1.set_ylabel('Average Sales')
    
    promo_effect['Customers'].plot(kind='bar', ax=ax2)
    ax2.set_title('Average Customers with/without Promo')
    ax2.set_ylabel('Average Customers')
    
    plt.tight_layout()
    plt.show()


# Analyze promo effectiveness by store type
def enhance_promo_analysis(df):
    
    promo_effect_by_store_type = df.groupby(['StoreType', 'Promo'])['Sales'].mean().unstack()
    promo_lift = (promo_effect_by_store_type[1] - promo_effect_by_store_type[0]) / promo_effect_by_store_type[0] * 100

    plt.figure(figsize=(10, 6))
    promo_lift.plot(kind='bar')
    plt.title('Promo Effectiveness by Store Type')
    plt.xlabel('Store Type')
    plt.ylabel('Sales Lift (%)')
    plt.show()

    # Suggest stores for promo deployment
    store_promo_effect = df.groupby('Store')[['Promo', 'Sales']].apply(lambda x: x[x['Promo'] == 1]['Sales'].mean() / x[x['Promo'] == 0]['Sales'].mean() - 1)
    top_stores_for_promo = store_promo_effect.nlargest(10)
    
    print("Top 10 stores where promos are most effective:")
    print(top_stores_for_promo)


def analyze_store_hours(df):
    # Group by DayOfWeek and calculate average sales
    daily_sales = df.groupby('DayOfWeek')['Sales'].mean().reset_index()
    
    # Create a bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='DayOfWeek', y='Sales', data=daily_sales)
    plt.title('Average Sales by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Sales')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.show()

    # Analyze open/closed patterns
    open_stores = df.groupby('DayOfWeek')['Open'].mean()
    print("Proportion of stores open by day of week:")
    print(open_stores)

    # Analyze sales for open stores
    open_sales = df[df['Open'] == 1].groupby('DayOfWeek')['Sales'].mean()
    print("\nAverage sales for open stores by day of week:")
    print(open_sales)


def analyze_weekday_open_stores(df):
    # Identify stores open on all weekdays
    weekday_open_stores = df[(df['Open'] == 1) & (df['DayOfWeek'].isin([1, 2, 3, 4, 5]))].groupby('Store')['DayOfWeek'].nunique()
    always_open_stores = weekday_open_stores[weekday_open_stores == 5].index

    # Calculate average weekday and weekend sales for all stores
    weekday_sales = df[df['DayOfWeek'].isin([1, 2, 3, 4, 5])].groupby('Store')['Sales'].mean()
    weekend_sales = df[df['DayOfWeek'].isin([6, 7])].groupby('Store')['Sales'].mean()

    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'Weekday Sales': weekday_sales,
        'Weekend Sales': weekend_sales,
        'Store Type': ['Always Open' if store in always_open_stores else 'Not Always Open' for store in weekday_sales.index]
    })

    # Melt the dataframe for easier plotting
    plot_data_melted = pd.melt(plot_data.reset_index(), id_vars=['Store', 'Store Type'], 
                               var_name='Day Type', value_name='Average Sales')

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Store Type', y='Average Sales', hue='Day Type', data=plot_data_melted)
    plt.title('Weekday vs Weekend Sales: Always Open Stores vs Others')
    plt.ylabel('Average Sales')
    plt.show()

    # Print summary statistics
    print("Summary Statistics:")
    print(plot_data.groupby('Store Type').agg({
        'Weekday Sales': ['mean', 'median'],
        'Weekend Sales': ['mean', 'median']
    }))

    # Analyze the difference in weekend vs weekday sales
    plot_data['Weekend_Weekday_Diff'] = plot_data['Weekend Sales'] - plot_data['Weekday Sales']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Store Type', y='Weekend_Weekday_Diff', data=plot_data)
    plt.title('Difference in Weekend vs Weekday Sales')
    plt.ylabel('Weekend Sales - Weekday Sales')
    plt.show()

    print("\nAverage difference in Weekend vs Weekday sales:")
    print(plot_data.groupby('Store Type')['Weekend_Weekday_Diff'].mean())



def analyze_assortment_effect(df):
    assortment_effect = df.groupby('Assortment')['Sales'].mean()
    
    plt.figure(figsize=(10, 6))
    assortment_effect.plot(kind='bar')
    plt.title('Average Sales by Assortment Type')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    plt.show()

def analyze_competition_distance(df_train):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_train, x='CompetitionDistance', y='Sales')
    plt.title('Sales vs Competition Distance')
    plt.xlabel('Competition Distance (meters)')
    plt.ylabel('Sales')
    plt.show()
    correlation = df_train['Sales'].corr(df_train['CompetitionDistance'])
    print(f'Sales vs Competition Distance (Correlation: {correlation:.2f})')


    df_train['CompetitionDistance_Binned'] = pd.cut(df_train['CompetitionDistance'], 
                                                    bins=[0, 1000, 5000, 10000, np.inf], 
                                                    labels=['0-1km', '1-5km', '5-10km', '>10km'])
    
    distance_sales = df_train.groupby('CompetitionDistance_Binned', observed=False)['Sales'].mean()

    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=distance_sales.index, y=distance_sales.values)
    plt.title('Average Sales by Competition Distance')
    plt.xlabel('Distance to Nearest Competitor')
    plt.ylabel('Average Sales')
    plt.show()


# analyze new competition distance 
def analyze_new_competitors(df_train):
    df_train['CompetitionOpen'] = np.where(df_train['CompetitionOpenSinceYear'] != 0, 1, 0)
    new_competitor_effect = df_train.groupby('CompetitionOpen')['Sales'].mean()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['No New Competitor', 'New Competitor'], y=new_competitor_effect.values)
    plt.title('Average Sales with and without New Competitors')
    plt.xlabel('New Competitor Status')
    plt.ylabel('Average Sales')
    plt.show()


