import requests

url = "http://localhost:7071/api/http_trigger_teamJ"


response1 = requests.post(url)

print(response1.text)