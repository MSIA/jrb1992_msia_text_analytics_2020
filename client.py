import requests

# Replace text in json file below, and run python client.py to see prediction result and probability.
response=requests.get("http://127.0.0.1:5000/predict", json={"text":"This tastes very sweet."})
print(response)
print(response.json())