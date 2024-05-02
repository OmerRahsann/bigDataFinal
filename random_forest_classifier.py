import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import redis
import yaml

# Load configuration and create a Redis connection
def load_config():
    try:
        with open("config.yaml", "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise

def get_redis_connection(config):
    try:
        connection = redis.Redis(
            host=config["redis"]["host"],
            port=config["redis"]["port"],
            db=0,
            decode_responses=True,
            username=config["redis"]["user"],
            password=config["redis"]["password"],
        )
        return connection
    except Exception as e:
        print(f"Error creating Redis connection: {e}")
        raise

def fetch_data_from_redis(redis_conn, limit=100):
    keys = redis_conn.keys('row:*')[:limit]
    data = []
    for key in keys:
        data.append(redis_conn.hgetall(key))
    return pd.DataFrame(data)

if __name__ == "__main__":
    config = load_config()
    redis_conn = get_redis_connection(config)
    df = fetch_data_from_redis(redis_conn, limit=100)  

    
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    # 'Diabetes_012' is the column to predict
    X = df.drop('Diabetes_012', axis=1)
    y = df['Diabetes_012']

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    # Feature importance visualization
    feature_importances = model.feature_importances_
    features = X.columns
    indices = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), feature_importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Relative Importance')
    plt.show()
