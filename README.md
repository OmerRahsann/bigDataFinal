# Data Finding
The dataset used in this project is the "Diabetes Health Indicators Dataset," available on Kaggle at this link. https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data This dataset includes a wide range of health indicators collected through surveys, making it ideal for studying factors influencing diabetes and related health conditions.

# Importance of the Data

Public Health Insights: It helps in identifying key health indicators associated with diabetes, which is essential for public health strategies and interventions.
Preventive Measures: Understanding which factors are closely linked with diabetes can aid in developing preventive measures.
Healthcare Optimization: The insights from this dataset can help healthcare providers optimize care plans and interventions for at-risk populations.

# Importing Data into Redis
The data was imported into Redis to leverage the high-performance data handling capabilities of Redis, which is particularly well-suited for managing and querying large datasets rapidly. Here’s a high-level overview of the steps taken:

Redis Import: The  data  imported into Redis using a Python script. Each record was stored as a hash in Redis, allowing for quick access and manipulation of individual health records.

Data Preparation: The dataset was cleaned and preprocessed using Python, ensuring that no duplicate or negative values and handling any missing values.

Reading Data from Redis
Data retrieval from Redis was executed via a Python script that connected to the Redis server. The script fetched data directly into a Pandas DataFrame for ease of manipulation and analysis. This process ensured that data handling was efficient and scalable, accommodating large volumes of data that could be fetched and analyzed on-the-fly.

# Machine Learning Models

Random Forest: This model was used to predict the likelihood of diabetes based on health indicators. It's beneficial for its ability to handle large datasets with many features and provides insights into feature importance, which helps in understanding which factors are most predictive of diabetes.

Logistic Regression: Used primarily to predict binary outcomes, logistic regression was applied to determine the risk factors for diseases like stroke or heart disease based on the health indicators. This model is advantageous for its interpretability and the ability to provide probabilities for outcomes, which are crucial for medical decision-making.

Gradient Boosting Machine (GBM): This powerful ensemble technique was used to improve predictions by combining multiple weak models into a robust one. GBM was particularly useful for handling varied types of data and improving prediction accuracy over single models.
