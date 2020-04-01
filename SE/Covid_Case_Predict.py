# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 09:38:32 2020

@author: BigDog
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

os.chdir(r"C:\Users\BigDog\Desktop\Python\Covid_19\covid_19_analysis\datasets")

#Names = ['dateRep', 'day', 'month', 'year', 'cases', 'deaths', 'countriesAndTerritories',\
#         'geoId', 'countryterritoryCode', 'popData2018']
drop_columns = ['geoId', 'countryterritoryCode', 'day', 'month', 'year']
target_country = "China"
# Import and format dataframe
covid19_df = pd.read_csv('COVID-19-geographic-disbtribution-worldwide-2020-03-31.csv', engine='python')
covid19_df['dateRep'] = pd.to_datetime(covid19_df['dateRep'], dayfirst=True)
covid19_df.drop(columns=drop_columns, inplace=True)
print(covid19_df.head())
# Create df for one country
country_df = covid19_df.loc[covid19_df.countriesAndTerritories == "China"].copy()
country_df.sort_values(by=['dateRep'], ascending=True, inplace=True)
print(country_df.head())
# Add cumulative columns for cases and deaths
country_df['Cum Cases'] = country_df['cases'].cumsum()
country_df['Cum Deaths'] = country_df['deaths'].cumsum()
print(country_df)
# Create column for days since x deaths
country_df['flag'] = np.where(country_df['Cum Cases'] > 100, 1, 0)
country_df['flag'] = np.where(country_df['Cum Cases'] > 100, country_df['flag'].cumsum(), 0)
print(country_df)
# Create series for days vs cases
print(country_df.columns)
X = country_df.iloc[:, 7:8].values
y = country_df.iloc[:, 5:6].values
print(X)
print(y)
# Fit linear relationship
lin_reg = LinearRegression()
lin_reg.fit(X, y)
# Fit polynomial relationship
poly_reg = PolynomialFeatures(degree = 9)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Cumulative Cases (Linear Regression)')
plt.xlabel('Days Since 100 Case')
plt.ylabel('Cases')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Cumulative Cases (Polynomial Regression)')
plt.xlabel('Days Since 100 Case')
plt.ylabel('Cases')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Cumulative Cases (Polynomial Regression)')
plt.xlabel('Days Since 100 Case')
plt.ylabel('Cases')
plt.show()


