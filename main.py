from flask import Flask, render_template, Response
from Face_mask_detector import VideoCamera
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
@app.route('/')

def index():
    # rendering webpage
    return render_template('index.html')

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # defining server ip address and port
    '''http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run(debug=True)'''
    app.run(host='0.0.0.0',port='5000', debug=True)

	