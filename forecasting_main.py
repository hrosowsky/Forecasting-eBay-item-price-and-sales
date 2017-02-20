import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
import forecasting_methods
#import pypyodbc



# Create the connection
#conn = pypyodbc.connect('DRIVER={SQL Server};''SERVER=DESKTOP-O6RSVA3\SQLEXPRESS;''DATABASE=ebay;')

# query db
sql = """

select  
condition_display_name as condition_display_name
, product_title as product_title
, cast(li_end_time as date) as date
, MAX(category_name) as category_name
, AVG(ss_current_price) as avg_price
, AVG(ss_current_price) + STDEV(ss_current_price) as upper_price
, AVG(ss_current_price) + STDEV(ss_current_price) as lower_price
, COUNT(*) as n_sold
from view_items_outliers_removed_final
where ss_selling_state = 'endedwithsales' 
group by condition_display_name, product_title, cast(li_end_time as date)
order by condition_display_name, category_name, product_title, date

"""
#df = pd.read_sql(sql, conn)
#df.to_pickle('ebay.df')
df = pd.read_pickle('ebay.df')
df['date'] = pd.to_datetime(df['date'])

# Filter for only New products
df = df[df['condition_display_name']=='New']

# Set index on product title
df = df.set_index(['product_title']).sort_index()

# Get some summary statistics for each product
avg_sold = df.n_sold.groupby(level=0).mean()
min_date = df.date.groupby(level=0).min()
max_date = df.date.groupby(level=0).max()
n_days = df.date.groupby(level=0).count()
category_name = df.category_name.groupby(level=0).max()

# Merge the summary statistics into one dataframe
df_summary = pd.concat([avg_sold, min_date, max_date, n_days, category_name]
                , keys=['avg_sold','min_date','max_date','n_days', 'category'], axis=1)

# Filter for product based on date range and minimum number of days where at least
# one product was sold
df_summary_filtered = df_summary[(min_date < '2016-07-01') 
                                & (max_date >= '2016-12-19')
                                & (n_days > 100)]


# We have 31 products. 
products = df[df.index.isin(df_summary_filtered.index)]

#Make sure all the products have the same date axis.
date_range = pd.date_range('2016-07-01', '2016-12-19')
products = products.groupby(level=0).apply(lambda x: x.set_index(['date']).reindex(date_range))

# Create a rolling average of the price/items sold for last 7 days.
products_rolling_mean = products[['avg_price','n_sold']].groupby(
                        level=0).apply(lambda x: pd.rolling_mean(x, window=7, min_periods=1))

# Fill forward and backward any `nan`.
products_rolling_mean = products_rolling_mean.groupby(level=0).apply(lambda x: x.fillna(method='ffill'))
products_rolling_mean = products_rolling_mean.groupby(level=0).apply(lambda x: x.fillna(method='bfill'))

# Note, we double check there are no nan's. 
assert products_rolling_mean.groupby(level=0).apply(lambda x: x.isnull().sum()).sum().sum() == 0

#Get Product-Category list.
products_category = products.reset_index()[['product_title', 'category_name']].drop_duplicates().dropna().set_index(['product_title'])


product_names = list(products_rolling_mean.index.levels[0].unique())

results_price = {}
results_sales = {}
for product in product_names:
    results_price[product] = forecasting_methods.linear_regression(products_rolling_mean.ix[product]['avg_price'])
    results_sales[product] = forecasting_methods.linear_regression(products_rolling_mean.ix[product]['n_sold'])
    

#Analysis



mse = pd.DataFrame([[None,None,None,None]]*len(product_names), index=product_names 
        , columns=['avg_price_train', 'avg_price_test', 'n_sold_train', 'n_sold_test'])

for product in product_names:
    mse.loc[product, 'avg_price_train'] = forecasting_methods.mean_squared_error(results_price[product][0]['train_y'], results_price[product][0]['train_yp'])
    mse.loc[product, 'avg_price_test'] = forecasting_methods.mean_squared_error(results_price[product][1]['test_y'], results_price[product][1]['test_yp'])
    mse.loc[product, 'n_sold_train'] = forecasting_methods.mean_squared_error(results_sales[product][0]['train_y'], results_sales[product][0]['train_yp'])
    mse.loc[product, 'n_sold_test'] = forecasting_methods.mean_squared_error(results_sales[product][1]['test_y'], results_sales[product][1]['test_yp'])
    mse.loc[product, 'category'] = products_category.ix[product].category_name

# Plotting
for product in product_names:
    forecasting_methods.plot(product,results_price[product][0], results_price[product][1]
        , results_sales[product][0], results_sales[product][1])









































##remove unsold items
#product_titles = []
#
#for product_title in df.index.levels[0].unique():
#    df_ = df.ix[product_title]
#    print(str(df_.))
#    if (df_.index.min() < '2016-07-01' and
#        df_.index.max() == '2016-12-20' and
#        len(df_) > 173):
#        product_titles.append(product_title)
        
#df_clean = df.loc[product_titles,:].reset_index().set_index(['product_title', 'date']).sort_index()

#for product_title in product_titles:
#    pass
#for category_name in df['category_name'].unique():
#    for condition_display_name in df['condition_display_name'].unique():
#        for product_title in df[']
#df = df.set_index(['product_title', 'date'])
#
#number_of_dates_per_product = df['n_sold'].groupby(level=0).count()
#min_dates_per_product = df['date'].groupby(level=0).min()
#max_dates_per_product = df['date'].groupby(level=0).max()
#avg_items_sold_per_day = df['n_sold'].groupby(level=0).sum()/(365/2)
# Removed from df any items which dont meet your satistifaction above


#remove products who have less than 10 sold items per week
#def clean_data(df):
    
#   mon = '2016-06-27'
#   tue = '2016-06-28'
#   wed = '2016-06-29'
 #  thur = '2016-06-30'
#   fri = '2016-07-01'
#   sat = '2016-07-02'
#   sun = '2016-07-03'

#for 