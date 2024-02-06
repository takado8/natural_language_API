import requests

url = "http://127.0.0.1:5131/"


def send_frame(bytes_data):
    files = {'file': ('filename', bytes_data)}
    # Send the request with the file data
    response = requests.post(url + 'frames', files=files)


async def get_recognition():
    print('requesting')
    response = requests.get(url + 'recognize')
    print(response.text)
    return response.text


if __name__ == '__main__':
    send_frame(None)