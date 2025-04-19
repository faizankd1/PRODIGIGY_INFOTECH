# PRODIGIGY_INFOTECH


# House Price Prediction using Linear Regression

## Overview

This project implements a **Linear Regression** model to predict house sale prices based on various house features. The dataset used is sourced from the Ames Housing dataset, which contains a variety of real-world features that can influence property values. The goal of this project is to build a predictive model that estimates sale prices of houses based on attributes such as living area, number of bedrooms, and overall quality.

The model's performance is evaluated using key metrics like the **R² Score** and **Mean Squared Error (MSE)**. Visual analysis is performed by comparing actual vs predicted sale prices in a scatter plot.

## Features

The following features from the dataset are used to train the model:

- GrLivArea: Above ground living area (in square feet)
- BedroomAbvGr: Number of bedrooms above ground
- FullBath: Number of full bathrooms
- OverallQual: Overall quality rating of the house (scale from 1 to 10)
- GarageCars: Number of cars that can fit in the garage
- TotalBsmtSF: Total square footage of basement area
- YearBuilt: Year the house was built

The target variable is salePrice, which represents the price at which the house was sold.

---

## Installation

### Prerequisites

To run this project, you need Python 3.x and the following Python libraries installed:

- pandas for data manipulation
- scikit-learn for machine learning algorithms
- matplotlib and seaborn for data visualization

You can install the required dependencies using the following pip command:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

## Getting Started

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/faizankd1/house-price-prediction.git
   ```

2. Prepare your dataset:
   - Download the `train.csv` dataset and place it in the project directory.
   - Ensure the file path in the script reflects the correct location of the dataset.

3. Run the Python script to train the model and generate predictions:

   ```bash
   python house_price_prediction.py
   ```

4. The script will:
   - Train a Linear Regression model using the training set.
   - Output the evaluation metrics:
     - R² Score: Measures the proportion of variance in the target variable (SalePrice) explained by the model.
     - **Mean Squared Error (MSE)**: Measures the average squared difference between the actual and predicted values.
   - Generate and display a scatter plot comparing actual vs predicted sale prices.

---

## Model Evaluation

### Key Metrics

After running the model, the following evaluation metrics will be shown in the output:

- R² Score: Indicates the proportion of variance in the target variable that is explained by the model. A higher R² score suggests better model performance.
  
- Mean Squared Error (MSE): A common loss function that measures the average squared difference between the predicted and actual values. Lower MSE values indicate better model accuracy.

Example output might look like:

```bash
R² Score: 0.81
Mean Squared Error: 125,734,589.92
```

This suggests that the model can explain 81% of the variance in sale prices, which is a solid performance for a linear regression model.

### Visualization

The script generates a scatter plot of **Actual vs Predicted Sale Prices**, which helps visualize how well the model predicts house prices:

- The red dashed line represents the ideal case where predicted values equal actual values (i.e., perfect predictions).
- The scatter points represent the predicted vs actual values, with points closer to the red line indicating more accurate predictions.

#### Actual vs Predicted Prices:

![Actual vs Predicted Prices](images/actual_vs_predicted.png)

*Note: Replace with the actual file path for your plot image, or upload it to the repository.*

---

## Analysis

### Key Insights

- Feature Importance: The most influential features on house prices are `OverallQual`, `GrLivArea`, and `TotalBsmtSF`. These features have strong correlations with sale prices.
  
- Model Limitations: While Linear Regression provides a good baseline, it may struggle to capture non-linear relationships and complex interactions between features. Thus, other models (e.g., Random Forest, Gradient Boosting) may yield better performance.

### Possible Improvements

- Feature Engineering: Creating additional features like `HouseAge = CurrentYear - YearBuilt` or incorporating neighborhood-related features could improve the model.
  
- Model Complexity: Testing more advanced models, such as **Random Forest** or **Gradient Boosting**, might improve prediction accuracy.

- Hyperparameter Tuning: Implementing grid search or random search for hyperparameter optimization can further refine the model’s performance.

---

## Future Work

- Model Enhancement: Explore ensemble methods like Random Forests or XGBoost for better accuracy.
- Web Deployment: Create a web application (using **Flask** or **Streamlit**) to allow users to input house features and receive predicted sale prices in real-time.
- Error Analysis: Conduct residual analysis to understand the model's prediction errors and identify areas of improvement.
- Advanced Models: Investigate other machine learning algorithms, including **Neural Networks** for potentially higher performance.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

