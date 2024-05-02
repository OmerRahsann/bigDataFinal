import redis
import yaml

# Load configuration from a YAML file
def load_config():
    try:
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
            print("Configuration loaded successfully.")
            return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise

# Create a Redis connection
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
        if connection.ping():
            print("Connected to Redis successfully.")
        return connection
    except Exception as e:
        print(f"Error creating Redis connection: {e}")
        raise

# Clean data directly in Redis
def clean_redis_data(redis_conn, pattern='row:*'):
    keys = redis_conn.keys(pattern)
    if not keys:
        print("No keys found matching the pattern.")
        return
    
    print(f"Found {len(keys)} keys matching the pattern.")
    seen_hashes = set()  # To identify duplicates

    for key in keys:
        data = redis_conn.hgetall(key)
        if not data:
            print(f"No data found for key: {key}")
            continue

        print(f"Processing key: {key}")
        for field, value in list(data.items()):
            try:
                # Handle negative values and nulls
                num_value = float(value)
                if num_value < 0:
                    redis_conn.hdel(key, field)
                    print(f"Negative value removed: {field} = {value} in {key}")
            except ValueError:
                if not value.strip():
                    redis_conn.hdel(key, field)
                    print(f"Null or empty value removed: {field} in {key}")

            value = value.strip().lower()
            if value != data[field]:
                redis_conn.hset(key, field, value)

        unique_signature = frozenset(data.items())
        if unique_signature in seen_hashes:
            redis_conn.delete(key)
            print(f"Duplicate removed: {key}")
        else:
            seen_hashes.add(unique_signature)

# Load configuration and clean data
if __name__ == "__main__":
    config = load_config()
    redis_conn = get_redis_connection(config)
    clean_redis_data(redis_conn)
