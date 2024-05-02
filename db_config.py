import redis
import yaml
import csv

def load_config():
  
    try:
        with open("config.yaml", "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise

def get_redis_connection(config):
   
    try:
        return redis.Redis(
            host=config["redis"]["host"],
            port=config["redis"]["port"],
            db=0,
            decode_responses=True,
            username=config["redis"]["user"],
            password=config["redis"]["password"],
        )
    except Exception as e:
        print(f"Error creating Redis connection: {e}")
        raise

def load_data_to_redis(csv_file_path, redis_conn):
   
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file) 
            pipeline = redis_conn.pipeline()
            count = 0

            for row in reader:
                # Generate a unique key for each row by appending the row count
                key = f"row:{count}"
                pipeline.hmset(key, row)  # Store the whole row as a hash
                count += 1
                if count % 1000 == 0:  
                    pipeline.execute()
                    pipeline = redis_conn.pipeline() 

            pipeline.execute()  
            print(f"All data inserted into Redis, total records: {count}")
    except Exception as e:
        print(f"Error loading data into Redis: {e}")
        raise


config = load_config()
redis_conn = get_redis_connection(config)
csv_file_path = r'C:\Users\omerr\Downloads\diabetes_012_health_indicators_BRFSS2015.csv'
load_data_to_redis(csv_file_path, redis_conn)
