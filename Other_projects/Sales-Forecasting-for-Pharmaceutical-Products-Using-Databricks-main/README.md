# Sales Forecasting for Pharmaceutical Products Using Databricks

## Project Overview

This project demonstrates the development of a **sales forecasting model** for pharmaceutical products using advanced analytics and machine learning techniques within the **Databricks** platform. Accurate sales forecasting is crucial for the pharmaceutical industry to optimize inventory management, production planning, and revenue projection. This project leverages historical sales data to build and evaluate predictive models that help forecast future sales for specific pharmaceutical products.

## Table of Contents
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Data Description](#data-description)
- [Methodology](#methodology)
- [Results](#results)
- [How to Use](#how-to-use)
- [Future Enhancements](#future-enhancements)
- [Acknowledgments](#acknowledgments)

---

## Motivation

The pharmaceutical industry faces numerous challenges related to demand variability, regulatory constraints, and the complexity of managing inventories. This project aims to address these challenges by:
- Building accurate forecasting models to predict product sales.
- Enabling better resource planning and cost management.
- Reducing stockouts and overstocking issues.

---

## Technologies Used

- **Databricks**: For data processing and model building.
- **Python**: Programming language for data analysis and machine learning.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib/Seaborn**: For data visualization.
- **scikit-learn**: For building and evaluating machine learning models.
- **Jupyter Notebooks**: For interactive development and experimentation.

---

## Features

- **Data Cleaning**: Handles missing values, inconsistent formats, and outliers.
- **Exploratory Data Analysis (EDA)**: Visualizes trends, seasonality, and correlations in sales data.
- **Feature Engineering**: Creates time-based and product-specific features for better model performance.
- **Machine Learning Models**: Implements predictive models such as ARIMA, XGBoost, and Random Forest.
- **Visualization**: Provides insightful plots for sales trends and forecasted results.
- **Automation**: Generates automated daily, weekly, and monthly forecasts.

---

## Data Description

The dataset includes historical sales data for pharmaceutical products, comprising the following:
- **Columns**:
  - `Product_Code`: Unique identifier for each product.
  - `Date`: Transaction date.
  - `Sales`: Number of units sold.
  - `Region`: Geographic region of the sale.
  - `Category`: Product category.
  - `Manufacturer`: Manufacturer information.

The dataset is stored in CSV format and is split into daily, weekly, and monthly sales files for detailed analysis.

---

## Methodology

1. **Data Preprocessing**:
   - Loaded and cleaned historical sales data.
   - Performed data normalization and aggregation to create structured datasets.

2. **Exploratory Data Analysis**:
   - Identified sales trends, seasonal patterns, and outliers using time-series plots and statistical techniques.

3. **Feature Engineering**:
   - Created lag features, rolling averages, and date-related features to improve model accuracy.

4. **Model Development**:
   - Implemented machine learning models including:
     - Linear Regression
     - ARIMA
     - XGBoost
     - Random Forest
   - Evaluated models using metrics such as RMSE, MAE, and R-squared.

5. **Visualization**:
   - Used Matplotlib and Seaborn to create dashboards and visualizations of sales trends and forecasts.

6. **Results Interpretation**:
   - Provided actionable insights based on the model’s predictions.

---

## Results

- Achieved **85% accuracy** in predicting monthly sales for the top-selling product category.
- Reduced forecast error (RMSE) by **15%** compared to baseline models.
- Identified seasonality trends that significantly impact sales in specific regions.

---

## How to Use

### Prerequisites
- Python 3.8 or higher.
- Required libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`.

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/naman1618/Sales-Forecasting-for-Pharmaceutical-Products-Using-Databricks.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Sales-Forecasting-for-Pharmaceutical-Products-Using-Databricks
   ```
3. Open the Jupyter notebook (`pharma.ipynb`) for step-by-step instructions on running the project.
4. Run the notebook or scripts on Databricks for complete execution.

---

## Future Enhancements

- **Integration with Real-Time Data Streams**: Enable real-time forecasting using streaming data pipelines.
- **Incorporate Additional Variables**: Include weather, economic indicators, and demographic data for improved accuracy.
- **Deploy Models**: Deploy the final model using cloud services like AWS or Azure for operational use.
- **Interactive Dashboards**: Develop dashboards using Tableau or Power BI for dynamic visualization.

---

## Acknowledgments

This project was inspired by the need to address inefficiencies in pharmaceutical sales forecasting. Special thanks to the University of Arizona and mentors for their guidance and support throughout this project.

---
