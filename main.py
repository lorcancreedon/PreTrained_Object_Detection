import cv2
import numpy as np
cap = cv2.VideoCapture('rtmp://192.168.137.1/live/drone')
##cap = cv2.VideoCapture(0)

##drone footage


# width height target
whT = 320
confidenceThreshold = 0.5
nmsThreshold = 0.3

classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT, = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshold:
                #w,h element number 3 and 4 in array
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

                ##lets apply nonmaximum suppression to remove overlapping boxes
    print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confidenceThreshold,nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,255),2)




#while loop that gets the frames of our webcam
while True:
    # tells us if cam image retrieved successfully
    success, img = cap.read()
    # the network only supports blob format so we convert cam into blob
    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(layerNames)
    #getting the output layers of the network
    #print(net.getUnconnectedOutLayers())
    #get the first element of i (200) and subtract 1 from it, this will give us the index
    #and we want to find the name at 199 from our layernames list (I.E. this should give us names of our output layers
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)

    outputs = net.forward(outputNames)
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)
    #print(outputs[0][0])
    findObjects(outputs,img)

    cv2.imshow('webcam', img)
    cv2.waitKey(1) #delay cam for 1 milisec
