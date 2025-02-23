import os
from dotenv import load_dotenv, find_dotenv, set_key

def setup_environment():
    """Set up environment variables"""
    env_file = find_dotenv()
    if not env_file:
        env_file = '.env'
    
    # Create .env file if it doesn't exist
    if not os.path.exists(env_file):
        open(env_file, 'a').close()
    
    # Load existing environment
    load_dotenv(env_file)
    
    # Set MongoDB URL if not already set
    mongo_url = os.getenv('MONGO_URL')
    if not mongo_url:
        mongo_url = 'mongodb+srv://Justine:Ju&tine2003@cluster0.rtxee8k.mongodb.net/'
        set_key(env_file, 'MONGO_URL', mongo_url)
        print(f"Added MONGO_URL to {env_file}")
    else:
        print("MONGO_URL already set")

if __name__ == "__main__":
    setup_environment()
