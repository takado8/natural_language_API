from flask import Flask, request
from keyword_recognition import KeywordRecognition

app = Flask(__name__)
recognition = KeywordRecognition()

frames = []


@app.route('/frames', methods=['POST'])
def gather_frames():
    request_data = request.files['file'].read()
    frames.append(request_data)
    return "OK"


@app.get('/recognize')
def recognize():
    result = recognition.recognize(frames)
    frames.clear()
    return str(result)


if __name__ == '__main__':
    app.run(debug=False, port=5131, use_reloader=False)
