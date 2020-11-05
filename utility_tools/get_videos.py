# import the necessary packages
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)
import datetime
import os.path
from os import path
import time
from utility_tools import line_de
from utility_tools import get_line
from config import config

lineState = True

tempPath = None
tempPathOut = None
imageName = None
lgCropHeight = 0
lgCropWidth = 0
dispImgTemp = None
dispImg = None
tempTime = None
detectVehicle = {}
hough_lines = []
lines = None

# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# #import tensorflow as tf
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")

ltx = 0
lty = 0
lbx = 0
lby = 0
rtx = 0
rty = 0
rbx = 0
rby = 0

tleft_lane = None
tright_lane = None
line = []

def yolov3(yolo_weights, yolo_cfg, coco_names):
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    clas = open(coco_names).read().strip().split("\n")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, clas, output_layers

net, clas, output_layers = yolov3(config.yoloWeightPath, config.yoloCfgPath, config.yoloClassPath)

class_names = [c.strip() for c in open(config.yoloClassPath).readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights(config.objWeightPath)
size = config.size
dt = datetime.datetime.now()
date = str(dt.year) + "_" + str(dt.month) + "_" + str(dt.day)
datepath = os.path.join(config.datepth, date)
out = None


max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = config.modelFileName
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

from _collections import deque
pts = [deque(maxlen=2) for _ in range(100000)]

counter = []
# img_array = []
frameIndex = 0

class VideoCamera(object):
    def __init__(self, capture_video):
        #capturing video
        self.video = cv2.VideoCapture(capture_video)
    
    def __del__(self):
        #releasing camera
        self.video.release()

    def getLineState(self):
        global lineState
        return lineState

    def changeState(self):
        global lineState
        if(lineState == True):
            lineState = False
        else:
            lineState = True
        return str(lineState)

    #sort taking x2 as reference
    def takeX2(self, elem):
        return elem[0]

    #sort taking Y1 as reference
    def takeY1(self, elem):
        return elem[3]

    #sort taking x1 as reference
    def takeX1(self, elem):
        return elem[2]

    def average_slope_intercept(self, lines):
        """
        Find the slope and intercept of the left and right lanes of each image.
            Parameters:
                lines: The output lines from Hough Transform.
        """
        global ltx
        global lty
        global lbx
        global lby
        global rtx
        global rty
        global rbx
        global rby
        listLine = lines.tolist()
        newList = listLine.copy()
        nList = []
        for each in newList:
            temp_list = []
            for item in each:
                for i in item:
                    temp_list.append(i)
            nList.append(temp_list)

        correctedList = []
        for element in nList:
            if(element[1] < element[3]):
                element[0], element[1], element[2], element[3] = element[2], element[3], element[0], element[1]
            correctedList.append(element)

        x2Sort = correctedList.copy()
        x2Sort.sort(key=self.takeX2)
        lbx = x2Sort[0][0]
        lby = x2Sort[0][1]
        rbx = x2Sort[-1][0]
        rby = x2Sort[-1][1]
        y1Sort = correctedList.copy()
        y1Sort.sort(key=self.takeY1)
        length = len(y1Sort)
        divide_index = length//2
        first_half = y1Sort[:divide_index]
        first_half.sort(key = self.takeX1)

        lty = first_half[0][3]
        ltx = first_half[0][2]
        rty = first_half[-1][3]
        rtx = first_half[-1][2]

        global tleft_lane
        global tright_lane
        if ((lbx - ltx) == 0):
            lbx += 1
        tleft_lines = []
        tleft_weights = []
        Tslope = (lby - lty) / (lbx - ltx)
        Tintercept = lty - (Tslope * ltx)
        Tlength = np.sqrt(((lby - lty) ** 2) + ((lbx - ltx) ** 2))
        tleft_lines.append((Tslope, Tintercept))
        tleft_weights.append((Tlength))
        tleft_lane = np.dot(tleft_weights,  tleft_lines) / np.sum(tleft_weights)  if len(tleft_weights) > 0 else None

        tright_lines = []
        tright_weights = []
        if((rbx - rtx) == 0):
            rbx += 1
        rslope = (rby - rty) / (rbx - rtx)
        rintercept = rty - (rslope * rtx)
        rlength = np.sqrt(((rby - rty) ** 2) + ((rbx - rtx) ** 2))
        tright_lines.append((rslope, rintercept))
        tright_weights.append((rlength))
        tright_lane = np.dot(tright_weights,  tright_lines) / np.sum(tright_weights)  if len(tright_weights) > 0 else None

        if ((Tslope < -14 and Tslope > -21) and (rslope > 40 and rslope < 62)):
            return tleft_lane, tright_lane
        else:
            return None

    def pixel_points(self, y1, y2, line):
        """
        Converts the slope and intercept of each line into pixel points.
            Parameters:
                y1: y-value of the line's starting point.
                y2: y-value of the line's end point.
                line: The slope and intercept of the line.
        """
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))

    def lane_lines(self, image, lines):
        """
        Create full lenght lines from pixel points.
            Parameters:
                image: The input test image.
                lines: The output lines from Hough Transform.
        """
        lane = self.average_slope_intercept(lines)
        if(lane is None):
            return None
        else:
            tleft_lane, tright_lane = lane
            y1 = image.shape[0]
            y2 = y1 * 0.4
            tleft_line  = self.pixel_points(y1, y2, tleft_lane)
            tright_line = self.pixel_points(y1, y2, tright_lane)
            return tleft_line, tright_line

        
    def draw_lane_lines(self, image, lines, color=[0, 0, 255], thickness=5):
        """
        Draw lines onto the input image.
            Parameters:
                image: The input test image.
                lines: The output lines from Hough Transform.
                color (Default = red): Line color.
                thickness (Default = 12): Line thickness. 
        """
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:           
                cv2.line(line_image, *line,  color, thickness)
        return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

    # Return true if line segments AB and CD intersect
    def intersect(self, A,B,C,D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

    def ccw(self, A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def frame_save(self, x, y, w, h, color, imagecp, track):
        global dispImg
        global lgCropWidth
        global lgCropHeight
        global dispImgTemp
        global tempTime
        global detectVehicle
        dispImg = None
        # saves image file
        imgCopy = imagecp.copy()
        crop = imgCopy[y:h,x:w]
        cropHeight, cropWidth, _ = crop.shape

        cubic = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        lab= cv2.cvtColor(cubic, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        crop=cv2.filter2D(final,-1,filter)

        cv2.rectangle(imagecp, (x, y), (w, h), color, 2)
        # Current Date and Time
        dt = datetime.datetime.now()
        # date = str(dt.year) + "_" + str(dt.month) + "_" + str(dt.day)
        tm = str(dt.strftime("%H:%M:%S"))
        global datepath
        vechiclepath = os.path.join(datepath, str(track.track_id))
        vechiclep = os.path.isdir(vechiclepath)
        if vechiclep != True:
            os.mkdir(vechiclepath)

        global tempPath
        global tempPathOut
        global imageName

        if (tempPath == None):
            tempPath = vechiclepath
            imageName = str(track.track_id)
            size = (1920,1080)
            tempPathOut = cv2.VideoWriter(vechiclepath + '/' +  str(track.track_id) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        elif(tempPath != vechiclepath):
            tempPathOut.release()
            tempPath = vechiclepath
            size = (1920,1080)
            tempPathOut = cv2.VideoWriter(vechiclepath + '/' +  str(track.track_id) + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
            imageName = str(track.track_id)
            lgCropHeight = 0
            lgCropWidth = 0
            dispImgTemp = None
        else:
            if(lgCropHeight < cropHeight or lgCropWidth < cropWidth):
                lgCropHeight = cropHeight
                lgCropWidth = cropWidth
                dispImgTemp = crop.copy()
                tempTime = tm
                cv2.imwrite(config.tempImagePath + imageName + ".png", dispImgTemp)
                detectVehicle[str(imageName)] = tempTime

        return(imagecp, crop, tm, vechiclepath, tempPathOut)

    def sframe(self, img, crop, tm, vechiclepath, tempPathOut):
        cv2.putText(img, tm, (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
        tempPathOut.write(img)
        cv2.imwrite(vechiclepath + "/" + "image-{}.png".format(frameIndex), img)
        cv2.imwrite(vechiclepath + "/" + "cropped-{}.png".format(frameIndex), crop)

    def perform_detection(self, net, imgLine, output_layers, w, h, confidence_threshold):
        blob = cv2.dnn.blobFromImage(imgLine, 1 / 255., (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Object is deemed to be detected
                if confidence > confidence_threshold:
                    center_x, center_y, width, height = list(map(int, detection[0:4] * [w, h, w, h]))

                    top_left_x = int(center_x - (width / 2))
                    top_left_y = int(center_y - (height / 2))

                    boxes.append([top_left_x, top_left_y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def draw_boxes(self, boxes, confidences, class_ids, cls, imgLine, confidence_threshold, NMS_threshold):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, NMS_threshold)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                cv2.rectangle(imgLine, (x, y), (x + w, y + h), (0,0,0), thickness= -1)
        return imgLine
        
    def get_frame(self):
       #extracting frames
        ret, img = self.video.read()
        global out
        if img is None:
            global tempPathOut
            tempPathOut.release()
            global imageName
            global dispImgTemp
            cv2.imwrite(config.tempImagePath + imageName + ".png", dispImgTemp)
            detectVehicle[str(imageName)] = tempTime
            imageName = None
            dispImgTemp = None
            import win32api
            win32api.MessageBox(0, 'Task Completed', 'Completed')
            out.release()
            self.video.release()
        
        else:
            cv2.ocl.setUseOpenCL(False)
            global datepath
            datep = os.path.isdir(datepath)
            if datep != True:
                os.mkdir(datepath)
            if out is None:
                # global datepath
                out = cv2.VideoWriter(datepath + '/' + 'test.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

            global hough_lines
            global frameIndex
            global line
            global lines
            if (frameIndex%10 == 0 or len(line)==0):
                imgLine = img.copy()
                imgLine= cv2.medianBlur(imgLine, 5)
                h, w, _ = imgLine.shape
                boxes, confidences, class_ids = self.perform_detection(net, imgLine, output_layers, w, h, 0.2)
                imgLine = self.draw_boxes(boxes, confidences, class_ids, clas, imgLine, 0.2, 0.4)
                gtLine = get_line.getLine(imgLine)
                if(gtLine is not None):
                    image, lg_x, lg_y, lg_w, lg_h = gtLine
                    test_image = [image]
                    hough_lines = line_de.execute(test_image, lg_x, lg_y, lg_w, lg_h)
                    # get line
                    for hlines in hough_lines:
                        lines = hlines
                        retline = self.lane_lines(img, lines)
                        if (retline is None):
                            continue
                        else:
                            line = retline

            imageCopy = img.copy()
            height, width, layers = img.shape
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, 416)

            t1 = time.time()

            boxes, scores, classes, nums = yolo.predict(img_in)

            classes = classes[0]
            names = []
            for i in range(len(classes)):
                names.append(class_names[int(classes[i])])
            names = np.array(names)
            converted_boxes = convert_boxes(img, boxes[0])
            features = encoder(img, converted_boxes)

            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                        zip(converted_boxes, scores[0], names, features)]

            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            tracker.predict()
            tracker.update(detections)

            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update >1:
                    continue
                bbox = track.to_tlbr()
                class_name= track.get_class()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                pts[track.track_id].append(center)

                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64/float(j+1))*2)
                    rec0 = [(bbox[0], bbox[3]), (bbox[2], bbox[3])]
                    rec1 = [(bbox[0], bbox[3]), (bbox[0], bbox[1])]
                    rec2 = [(bbox[2], bbox[1]), (bbox[0], bbox[1])]
                    rec3 = [(bbox[2], bbox[1]), (bbox[2], bbox[3])]
                    rLines = [rec0, rec1, rec2, rec3]
                    for rLine in rLines:
                        if(len(line)!=0):
                            if (self.intersect(rLine[0], rLine[1], line[0][0], line[0][1]) or self.intersect(rLine[0], rLine[1], line[1][0], line[1][1])):
                                cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0, 0, 255), 2)
                                imgVec, crop, tm, vechiclepath, tempPathOut = self.frame_save(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), (0, 0, 255), imageCopy, track)
                                self.sframe(imgVec, crop, tm, vechiclepath, tempPathOut)
                                break
                            elif len(lines)==1 and self.intersect(rLine[0], rLine[1], line[0], line[1]):
                                cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0, 0, 255), 2)
                                imgVec, crop, tm, vechiclepath, tempPathOut = self.frame_save(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), (0, 0, 255), imageCopy, track)
                                self.sframe(imgVec, crop, tm, vechiclepath, tempPathOut)
                                break

                    frameIndex += 1
                    lineDisplay = self.getLineState()
                    if(lineDisplay == True):
                        img = self.draw_lane_lines(img, line)

            out.write(img)

            # encode OpenCV raw frame to jpg and displaying it
            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()