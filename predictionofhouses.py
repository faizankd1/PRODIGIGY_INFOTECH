# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load the Dataset
# Use raw string format or proper slashes
df = pd.read_csv(r"C:\prodigyinfotech\train.csv")

# 3. Select Features and Target
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 
            'OverallQual', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
target = 'SalePrice'

X = df[features]
y = df[target]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict on Test Set
y_pred = model.predict(X_test)

# 7. Evaluate the Model
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# 8. Plot Actual vs Predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()
