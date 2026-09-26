import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENALEX_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENALEX_API_KEY is not set in .env")

BASE_URL = "https://api.openalex.org"


def test_api():
    url = f"{BASE_URL}/works"

    params = {
        "api_key": API_KEY,
        "search": "natural language processing",
        "per-page": 5,
    }

    response = requests.get(url, params=params, timeout=30)

    response.raise_for_status()

    data = response.json()

    print("API connection successful")
    print("Total matching works:", data["meta"]["count"])

    for work in data["results"]:
        print("-", work.get("display_name"))


if __name__ == "__main__":
    test_api()