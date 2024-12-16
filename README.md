# Food Delivery

## Objective
Predicting the delivery time for food orders is a critical challenge for food delivery services like **Zomato** and **Swiggy**. An accurate prediction of delivery time improves customer satisfaction by setting realistic expectations and helps optimize delivery operations by efficiently allocating delivery resources.  

The project focuses on leveraging historical data to build a predictive model that calculates delivery time based on key factors such as:
- **Distance** between the restaurant (pickup point) and the customer's location (delivery point).
- Delivery patterns observed in the past for similar distances and conditions.

This initiative aims to enhance the **efficiency**, **reliability**, and **accuracy** of delivery time predictions, thus addressing a fundamental challenge faced by food delivery services.

---

## Key Features
1. **Distance Calculation**  
   Uses the Haversine formula to compute the distance between the restaurant and the delivery location based on latitude and longitude coordinates.  

2. **Data Cleaning and Transformation**  
   - Handles missing values and scales numerical data.
   - Encodes categorical variables using **Ordinal Encoding**.  

3. **Predictive Model Training**  
   - Trains and evaluates multiple regression models such as **Linear Regression**, **Lasso**, **Ridge**, and **ElasticNet**.
   - Selects the best-performing model for delivery time prediction.

4. **New Input Prediction**  
   Accepts custom inputs for predicting delivery time and provides real-time results.

---

## Installation

### 1. Install Required Packages
To install the necessary dependencies, use the following command:  
```bash
pip install -r requirements.txt
```

# Process Followed
## Data Cleaning Process

The `dataCleaning` class is designed to preprocess and clean the dataset for further analysis and modeling. Below are the key steps involved in the data cleaning process:

### Steps:

1. **Dropping Irrelevant Columns**  
   - Removed columns like `ID` and `Delivery_person_ID` as they are not relevant to predictive modeling.

2. **Handling Missing Values**  
   - Rows with three or more missing values are identified and removed.
   - Imputed specific missing values with appropriate strategies:
     - `Festival`: Filled with "No" (the most frequent value).
     - `day_quaters`: Filled with "night" (common ordering time).
     - `Delivery_person_Ratings`: Imputed with the mean rating, rounded to one decimal place.
     - `Delivery_person_Age`: Imputed with the mean age, rounded to the nearest integer.
     - `multiple_deliveries`: Imputed with the mean value.

3. **Creating a New Feature**  
   - Introduced a new column `day_quaters`, which divides the day into four segments based on the order time (`Time_Orderd`):
     - `morning` (05:00 - 10:00)
     - `late morning` (10:00 - 14:00)
     - `afternoon` (14:00 - 19:00)
     - `night` (19:00 onwards)

4. **Dropping Redundant Columns**  
   - Removed columns such as `Time_Orderd`, `Time_Order_picked`, and `Order_Date` after extracting relevant features.

5. **Final Cleanup**  
   - Dropped any remaining rows with missing values to ensure a clean dataset.

### Outcome:

This process ensures the dataset is clean, consistent, and ready for further analysis or modeling, with minimal noise and maximum reliability.

## Data Transformation Process

The `dataTransformation` class is designed to further preprocess the cleaned dataset by adding new features, scaling, encoding, and splitting it for training and testing. Below are the key steps involved in the data transformation process:

---

### Steps:

1. **Calculating Distance**  
   - Used the Haversine formula to calculate the distance between the restaurant and the delivery location.  
   - The formula considers the latitude and longitude of both locations and the Earth's radius (6371 km) to compute the distance accurately.

   **Haversine Formula:**  
   \[
   a = \sin^2\left(\frac{\Delta\text{lat}}{2}\right) + \cos(\text{lat1}) \cdot \cos(\text{lat2}) \cdot \sin^2\left(\frac{\Delta\text{lon}}{2}\right)
   \]
   \[
   c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1-a}\right)
   \]
   \[
   \text{Distance} = R \cdot c
   \]  
   - Added a new column `distance` to the dataset representing the calculated distance.

---

2. **Splitting the Dataset**  
   - Target variable: `Time_taken (min)`  
   - Features (`X`) are all columns excluding the target variable, while the target (`Y`) contains only the `Time_taken (min)` column.

---

3. **Categorical and Numerical Columns**  
   - Identified categorical and numerical columns for preprocessing:
     - **Categorical Columns**: Include object-type features such as `City`, `Festival`, `Road_traffic_density`, etc.
     - **Numerical Columns**: Include features such as latitude, longitude, distance, etc.

---

4. **Pipelines for Preprocessing**  

   - **Numerical Pipeline**:  
     - Imputed missing values using the **mean** strategy.  
     - Scaled numerical values using **StandardScaler**.

   - **Categorical Pipeline**:  
     - Imputed missing values using the **most frequent** strategy.  
     - Encoded categorical features using **OrdinalEncoder** with predefined mappings for:
       - `City`
       - `Festival`
       - `Road_traffic_density`
       - `Type_of_order`
       - `Type_of_vehicle`
       - `Weather_conditions`
       - `day_quaters`
     - Scaled encoded values using **StandardScaler**.

   - Combined both pipelines using `ColumnTransformer` for consistent preprocessing.

---

5. **Splitting into Train and Test Sets**  
   - Split the preprocessed dataset into training (70%) and testing (30%) subsets using `train_test_split`.  
   - Applied preprocessing pipelines to transform both `X_train` and `X_test`.

---

### Outcome:

This process prepares the dataset by calculating distances, imputing missing values, encoding categorical features, scaling numerical values, and splitting the data into train-test sets. The transformed dataset is ready for training machine learning models.

## Model Training Process

The `modelTraining` class is designed to train, evaluate, and use regression models to make predictions on the transformed dataset. Below are the key steps involved in the model training and prediction process:

---

### Steps:

1. **Model Training**  
   - Trained multiple regression models on the preprocessed dataset:
     - **Linear Regression**
     - **Lasso Regression**
     - **Ridge Regression**
     - **ElasticNet Regression**

   - The models were trained using `X_train` (features) and `y_train` (target variable).

   - Evaluated each model's performance on the test data (`X_test`, `y_test`) using the following metrics:
     - **Mean Absolute Error (MAE):** Measures the average magnitude of errors in predictions.
     - **Mean Squared Error (MSE):** Penalizes larger errors more heavily.
     - **Root Mean Squared Error (RMSE):** Square root of MSE for error scale comparison.
     - **R² Score:** Represents the proportion of variance explained by the model.

   - Stored the performance metrics (`r2_score`) for comparison across models.

---

2. **Evaluation and Logging**  
   - For each model:
     - Printed the model's name.
     - Displayed its performance metrics (RMSE, MAE, R² score).
     - Tracked the results for further analysis or reporting.

---

3. **Best Model**  
   - The method currently returns the **last trained model**. This can be modified to return the best-performing model based on `R² score` or another metric.

---

4. **Prediction**  
   - Created a `modelPredict` method for generating predictions on new inputs.  
   - Accepts `new_input` (a list or array of feature values) and the trained model.  
   - Outputs the prediction result.

---

5. **Integration with Data Preprocessing**  
   - Cleaned and transformed the dataset using the `dataCleaning` and `dataTransformation` classes before model training.  
   - These steps include:
     - Removing inconsistencies.
     - Adding a `distance` feature using the Haversine formula.
     - Preprocessing data with pipelines.
     - Splitting into train and test sets.

---

6. **Testing New Inputs**  
   - Provided a sample input:
     ```
     new_input = [[36.0, 3, 30.327968, 78.046106, 30.397968, 78.116106, 1, 1, 2, 1, 1, 3.0, 2, 1, 1, 10.280582]]
     ```
   - Used the trained model to predict the `Time_taken (min)` for this input.  
   - Printed the prediction result for verification.

---

### Outcome:

This process integrates data cleaning, transformation, training, evaluation, and prediction in a pipeline. The model's performance is measured, and predictions can be made on new data, ensuring a streamlined approach to building regression models for food delivery time prediction.









