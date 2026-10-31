# Step 1: Load and Explore the Data

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm

df = pd.read_csv("https://raw.githubusercontent.com/cybervaults/Bi/refs/heads/main/practical6.csv")

print(df.head())
print(df.describe().round(1))
print("Missing values:", df.isna().sum().sum())


# Step 2: Check the Correlation

corr = df.drop(columns="store").corr()
plt.figure(figsize=(6, 4.5))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation matrix")
plt.show()


# Step 3: Simple Linear Regression

X1 = df[["ad_spend_k"]]
y = df["weekly_sales_k"]
simple = LinearRegression().fit(X1, y)

print("Intercept:", round(simple.intercept_, 2))
print("Slope (sales per 1k ad spend):", round(simple.coef_[0], 2))
print("R-squared:", round(simple.score(X1, y), 3))


# Step 4: Multiple Linear Regression

features = ["ad_spend_k", "footfall", "avg_discount_pct"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print("R-squared :", round(r2_score(y_test, pred), 3))
print("MAE       :", round(mean_absolute_error(y_test, pred), 2))
print("RMSE      :", round(np.sqrt(mean_squared_error(y_test, pred)), 2))
print(pd.DataFrame({"Feature": features,
                    "Coefficient": model.coef_.round(3)}))
print("Intercept:", round(model.intercept_, 2))


# Step 5: Statistical Summary using statsmodels

Xc = sm.add_constant(df[features])
ols = sm.OLS(y, Xc).fit()
print(ols.summary().tables[1])
print("Adjusted R-squared:", round(ols.rsquared_adj, 3))


# Step 6: Residual Analysis

res = y_test - pred
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].scatter(pred, res, alpha=0.6)
ax[0].axhline(0, color="red")
ax[0].set_xlabel("Predicted sales")
ax[0].set_ylabel("Residual")
ax[0].set_title("Residuals vs Predicted")
ax[1].hist(res, bins=12)
ax[1].set_title("Residual Distribution")
plt.tight_layout()
plt.show()


# Step 7: Predict Weekly Sales for New Data

new = pd.DataFrame([[40, 3000, 10]], columns=features)
print("Predicted weekly sales (thousand Rs.):",
      round(model.predict(new)[0], 1))
