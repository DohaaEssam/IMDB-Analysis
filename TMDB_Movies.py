import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("tmdb-movies.csv", index_col=0)
x = df.genres.str.get_dummies(sep="|")
df[["month", "day", "year"]] = df["release_date"].str.split("/", expand=True)
y = df[["runtime", "vote_average", "month", "release_year", "budget_adj", "revenue_adj"]]
new_df = [y, x]
TMDB_df = pd.concat(new_df, axis=1, join='inner')
a = TMDB_df['runtime'].mean()
TMDB_df['runtime'].fillna(a, inplace=True)

b = TMDB_df['vote_average'].mean()
TMDB_df['vote_average'].fillna(b, inplace=True)

TMDB_df.drop_duplicates(inplace=True)
TMDB_df.dropna(inplace=True)
print(TMDB_df)

le = LabelEncoder()
month = le.fit_transform(TMDB_df['month'])
TMDB_df['month'] = month
year = le.fit_transform(TMDB_df['release_year'])
TMDB_df['release_year'] = year

df_min_max_scaled = TMDB_df.copy()
for column in df_min_max_scaled.columns:
    df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (
                df_min_max_scaled[column].max() - df_min_max_scaled[column].min())

x = df_min_max_scaled
y = df_min_max_scaled['revenue_adj'] - df_min_max_scaled['budget_adj']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
model = regressor.fit(X_train, Y_train)
print("Y_Intercept: ")
print(model.intercept_)
print("The Coefficient: ")
print(model.coef_)
y_pred = regressor.predict(Y_test)
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
plt.scatter(Y_test, y_pred)
plt.show()
X_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': X_test, 'Predicted': X_pred})
plt.scatter(X_test, X_pred)
plt.show()
