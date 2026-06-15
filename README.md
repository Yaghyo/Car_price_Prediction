# 🚗 Car Price Prediction: Ensemble Regression Techniques

## 📖 Comprehensive Overview
This data science project focuses on accurately predicting the depreciation and current market value of used cars. Using the extensive **CarDekho dataset**, this project highlights the critical steps of the data science workflow: rigorous data cleaning, feature engineering, exploratory data analysis (EDA), and the application of powerful ensemble machine learning models. 

By analyzing various physical, historical, and mechanical attributes of a vehicle, the system outputs a highly accurate estimate of its selling price.

## 📊 Dataset Overview (`cardekho_imputated.csv`)
The project utilizes a rich, pre-processed dataset encompassing multiple facets of a vehicle's history and specifications. 

### **Target Variable:**
* `selling_price`: The continuous target variable representing the price of the car in INR.

### **Key Features (Predictors):**
* **Identifiers:** `car_name`, `brand`, `model`
* **Historical/Usage Data:**
  * `vehicle_age`: Age of the car in years (derived feature).
  * `km_driven`: Total distance the car has been driven.
* **Categorical/Sales Data:**
  * `seller_type`: The entity selling the car (e.g., Individual, Dealer, Trustmark Dealer).
  * `fuel_type`: The type of fuel required (Petrol, Diesel, CNG, LPG).
  * `transmission_type`: Gearbox configuration (Manual, Automatic).
* **Mechanical Specifications:**
  * `mileage`: Fuel efficiency metric.
  * `engine`: Engine displacement/capacity in cubic centimeters (CC).
  * `max_power`: Maximum power output of the engine in Brake Horsepower (bhp).
  * `seats`: Passenger capacity of the vehicle.

## 🔬 Data Processing & Feature Engineering
Raw real-world data is rarely ready for modeling. This project heavily emphasizes the preprocessing phase to prepare the CarDekho dataset:
* **Data Imputation:** Addressed missing values in critical mechanical columns (like engine and max_power) using statistical imputation techniques (e.g., mean/median for continuous variables, mode for categorical).
* **Feature Extraction:** Cleaned string-based columns to extract pure numerical values (e.g., converting strings like "18.5 kmpl" to a float `18.5`).
* **Categorical Encoding:** Transformed nominal and ordinal features (Fuel Type, Seller Type, Transmission) into machine-readable mathematical formats using techniques like One-Hot Encoding and Label Encoding.

## 🤖 Ensemble Modeling Strategy
Instead of relying on basic single estimators like Linear Regression, this project leverages **Ensemble Learning** to boost predictive accuracy, handle non-linear relationships, and reduce variance:
* **Random Forest Regressor:** Uses the "Bagging" (Bootstrap Aggregating) technique. It builds multiple decision trees on random subsets of the data and averages their predictions. This makes the model highly robust against overfitting and less sensitive to outliers in features like `km_driven` or `selling_price`.
* **AdaBoost Regressor:** Uses the "Boosting" technique. It sequentially builds models, where each new model pays more attention to the instances (cars) that the previous models predicted poorly, iteratively refining the prediction of complex or rare car prices.

## 📈 Evaluation Metrics
The models' performances are rigorously compared using standard regression metrics to ensure reliability:
* **R-Squared ($R^2$):** Used to determine the proportion of variance in the car price that can be explained by the given features.
* **Mean Absolute Error (MAE):** Calculates the average magnitude of the errors in a set of predictions, providing a clear dollar-value (or Rupee-value) average of how far off the predictions are.
* **Root Mean Squared Error (RMSE):** Penalizes larger errors more significantly, ensuring the model is reliable for high-value/luxury car predictions where the absolute error could be much larger.

## 🛠️ Detailed Tech Stack
* **Environment:** Jupyter Notebook (`RandomF_Adaboost_regression.ipynb`)
* **Data Manipulation & Cleaning:** Pandas, NumPy
* **Machine Learning & Preprocessing:** Scikit-Learn (Ensemble Methods, Evaluation Metrics, Preprocessing modules)
* **Data Visualization:** Seaborn, Matplotlib