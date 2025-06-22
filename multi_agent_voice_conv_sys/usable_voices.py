import os
import requests
from dotenv import load_dotenv

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

url = "https://api.deepgram.com/v1/speak/voices"
headers = {
    "Authorization": f"Token {DEEPGRAM_API_KEY}"
}

response = requests.get(url, headers=headers)

print("Status Code:", response.status_code)
print("Raw Response Text:\n", response.text)


# NOT FOR IMPLEMENTATION, JUST FOR DEEPGRAM TESTING AND DEBUGGING