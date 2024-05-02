import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import redis
import yaml
import matplotlib.pyplot as plt


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

# Main data handling and machine learning pipeline
if __name__ == "__main__":
    config = load_config()
    redis_conn = get_redis_connection(config)
    df = fetch_data_from_redis(redis_conn, limit=100)

    # Convert data to numeric, handle missing data
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    # Select features and target
    features = ['HighBP', 'Smoker', 'PhysHlth', 'HvyAlcoholConsump', 'Veggies']
    X = df[features]
    y = df['GenHlth']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train Gradient Boosting Classifier
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbm.fit(X_train, y_train)

    # Predict and evaluate the model
    predictions = gbm.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))

    # Plot feature importances
    feature_importances = gbm.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importances, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance in Gradient Boosting Model')
    plt.show()
