from pymongo import MongoClient
from dotenv import load_dotenv
import os

def test_mongodb():
    try:
        load_dotenv()
        mongo_url = os.getenv('MONGO_URL')
        if not mongo_url:
            print("Error: MONGO_URL not found in .env file")
            return False
            
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        client.server_info()
        print("Successfully connected to MongoDB!")
        return True
        
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return False

if __name__ == "__main__":
    test_mongodb()
