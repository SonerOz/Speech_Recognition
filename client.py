import requests

URL = "http://127.0.0.1/predict" # Change the ip adress to aws public ip after launching it to AWS
AUDIO_TEST = "test/cat.wav"


if __name__ == "__main__":
    
    audio_file = open(AUDIO_TEST, "rb")
    values = {"file": (AUDIO_TEST, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()
    
    print(f"Predicted keyword is: {data['keyword']}")    