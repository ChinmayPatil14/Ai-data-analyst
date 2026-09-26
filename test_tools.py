import pandas as pd
from tools import describe_data, correlation, filter_data, group_by_agg, plot_column

df = pd.read_csv("data/sales.csv")

print("---- describe_data ----")
print(describe_data(df))

print("\n---- correlation (price, quantity) ----")
print(correlation(df, "price,quantity"))

print("\n---- filter_data (region == North) ----")
print(filter_data(df, "region == 'North'"))

print("\n---- group_by_agg (region -> mean price) ----")
print(group_by_agg(df, "region", "price", "mean"))

print("\n---- plot_column (histogram of price) ----")
fig = plot_column(df, "price", "histogram")
print(type(fig))
fig.show()