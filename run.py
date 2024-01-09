import requests

image_path = 'tesT_her.jpeg'

api_url = 'http://127.0.0.1:5000/predictimage'

with open(image_path, 'rb') as file:
    image_data = file.read()

files = {'image': (image_path, image_data, 'multipart/form-data')}

response = requests.post(api_url, files=files)

if response.status_code == 200:
    print("Prediction Result:", response.text)
else:
    print("Failed to get predictions.")
