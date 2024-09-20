import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_promo_distribution(train, test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    train['Promo'].value_counts().plot(kind='pie', ax=ax1, autopct='%1.1f%%')
    test['Promo'].value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax1.set_title('Promo Distribution in Train Set')
    ax2.set_title('Promo Distribution in Test Set')
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

    # Analyze sales by day of week
    plt.figure(figsize=(12, 6))
    sns.barplot(x='DayOfWeek', y='Sales', data=df)
    plt.title('Sales Distribution by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Sales')
    plt.show()

def analyze_seasonal_sales(df):
    df['Month'] = df['Date'].dt.month
    monthly_sales = df.groupby('Month')['Sales'].mean()
    
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='bar')
    plt.title('Average Monthly Sales')
    plt.xlabel('Month')
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

def analyze_competition_distance(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='CompetitionDistance', y='Sales')
    plt.title('Sales vs Competition Distance')
    plt.xlabel('Competition Distance (meters)')
    plt.ylabel('Sales')
    plt.show()
    correlation = df['Sales'].corr(df['CompetitionDistance'])
    print(f'Sales vs Competition Distance (Correlation: {correlation:.2f})')

def analyze_new_competitors(df):
    df['CompetitorAge'] = df['Date'].dt.year - df['CompetitionOpenSinceYear']
    df['CompetitorAge'] = df['CompetitorAge'].clip(lower=0)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='CompetitorAge', y='Sales')
    plt.title('Average Sales vs Competitor Age')
    plt.xlabel('Years Since Competitor Opened')
    plt.ylabel('Average Sales')
    plt.show()