import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv("amazon sales.csv")

print(df.isna().sum())

df = df.fillna(df.mean())
df['discount_percentage'] = df['discount_percentage'].apply(lambda x: float(x.strip('%')))
df["discounted_price"] = df["discounted_price"].str.replace(",", "").astype(float)
df["actual_price"] = df["actual_price"].str.replace(",", "").astype(float)
df["rating_count"] = df["rating_count"].str.replace(",", "")


print(df.head())


X = df[["discounted_price", "discount_percentage",  "rating", "rating_count"]]
y = df["actual_price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


regressor = LinearRegression()


regressor.fit(X_train, y_train)


with open("model.pkl", "wb") as file:
    pickle.dump(regressor, file)


y_pred = regressor.predict(X_test)
result_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
print(result_df.head())