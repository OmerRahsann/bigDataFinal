import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import redis
import yaml
import matplotlib.pyplot as plt

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

    # Convert data to numeric, handle missing data
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    # Prepare data for modeling
    X = df.drop(['Stroke'], axis=1) 
    y = df['Stroke']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42) 
    lr_model.fit(X_train, y_train)

    # Predict and evaluate the model
    predictions = lr_model.predict(X_test)
    probabilities = lr_model.predict_proba(X_test)[:, 1]  

    print("Classification Report:\n", classification_report(y_test, predictions))
    print("ROC AUC Score:", roc_auc_score(y_test, probabilities))

    # Feature importance visualization based on coefficients
    features = X.columns
    feature_importances = lr_model.coef_[0]
    plt.figure(figsize=(12, 8))
    sorted_indices = feature_importances.argsort()
    plt.barh(features[sorted_indices], feature_importances[sorted_indices], color='skyblue')
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance in Logistic Regression Model')
    plt.show()
