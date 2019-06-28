# linear-regression
linear_regression  in machine learning
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# load the data into dataframe
df = pd.read_csv("D:\DATASCIENCE/3-linear-reg/canada_per_capita_income.csv")
df

# plot a scatter plot
plt.xlabel('years')
plt.ylabel('canada_per_capita_income (US$)')
plt.scatter(df.year, df.income, color='red', marker='*')

# the distribution is linear and can use Liner regression  logic.
reg = LinearRegression()
reg.fit(df[['year']], df.income)  # training



# ----------------------------------------------------------------------------------

reg.predict([[2019]]) #40460.22901919
reg.predict([[2020]]) #41288.69409442
reg.predict([[2021]]) #42117.1591696
reg.predict([[2022]]) #42945.6242448
reg.predict([[2023]]) #43774.08932009
reg.predict([[2024]]) #44602.55439531


#displaying the scatter plot

plt.xlabel('year')
plt.ylabel('canada_per_capita_income (US$)')
plt.scatter(df.year, df.income, color='red', marker='*')
plt.plot(df.year, reg.predict(df[['year']]), color='blue')
