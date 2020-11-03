from flask import Flask, render_template, request, Response, flash
import cv2
# from utility_tools import get_videos
from utility_tools.get_videos import VideoCamera
from utility_tools import get_line
from utility_tools import line_de
import os
from utility_tools.get_videos import lineState

VIDEO_FOLDER = 'E:/Series/flaskApp/data/video/temporary/'
# ALLOWED_EXTENSIONS = {'mp4'}
app = Flask(__name__)
app.config['videoPath'] = VIDEO_FOLDER

videoShow = 0
capture_video = None
dispImgArray = []

def video(Video_file):
	filename = Video_file.filename
	Video_file.save(os.path.join(app.config['videoPath'], filename))
	filePath = os.path.join(app.config['videoPath'], filename)

	global capture_video
	# print('\n')
	# print(filePath)
	# capture_video = cv2.VideoCapture(filePath)
	capture_video = filePath
	# return capture_video

@app.route('/')
def index():
	return render_template("index.html")

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/predict',methods=['GET','POST'])
def predict():
	Video_file = request.files['videoFile']
	if Video_file.filename !='':
		video(Video_file)
		if(lineState == True):
			lineStateValue = "Enabled"
		else:
			lineStateValue = "Disabled"
		return render_template("videoDisplay.html", lineState = lineStateValue)
		
@app.route('/displayVideo')
def displayVideo():
	# cv2.ocl.setUseOpenCL(False)
	lineDisplay = True
	global capture_video
	return Response(gen(VideoCamera(capture_video)),
					mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/displayImage')
def displayImage():
	from utility_tools.get_videos import detectVehicle
	return render_template('showImages.html', detectVehicle = detectVehicle)


@app.route('/lineState',methods=['GET','POST'])
def StateLine():
	from utility_tools.get_videos import lineState
	lineState = VideoCamera.changeState(lineState);
	return str(lineState)

if __name__ == "__main__":
	app.run(debug=True)