<<<<<<< HEAD
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
#for reducing frame size
ds_factor = 0.9

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release(0)
    
    def get_frame(self):
        frame_status, frame = self.video.read()
        #resize frame
        frame = cv2.resize(frame,None, fx = ds_factor, fy = ds_factor, interpolation = cv2.INTER_AREA)
        #convert captured frame into gray-scale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #equalize image hist to increase contrast and increase feauture detection accuracy
        gray_frame = cv2.equalizeHist(gray_frame)
        #detect the face frame rectangle(top left, bottom down coordinates)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        wearing_mask = None
        for (x, y, w, h) in faces:
            #increase nose detection accuracy by making the face the region of interest 
            face_frame = frame[y : y + h, x : x + w]
            resized_frame = cv2.resize(face_frame, (256, 256))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            #detect nose position
            nose = nose_cascade.detectMultiScale(face_frame, 1.3, 5)
    
            no_of_noses = len(nose)
            if no_of_noses == 0:
                wearing_mask = True
            else:
                wearing_mask = False
            #set frame color according to mask detected or not
            if wearing_mask == True:
                frame_color = (0, 255, 0)
            elif wearing_mask == False:
                frame_color = (0, 0, 255)
            else:
                frame_color = (255, 0, 0)
            center = (x + w//2, y + h//2)
            #frame the face
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, frame_color, 4)
            #add appropraiate label to frame
            if wearing_mask:
                face_label = "Great! You are a responsible citizen"
            
            else:
                face_label = "Go wear a mask!!"
                #face_label = "Don't risk your and others life.Go wear a mask!"
            frame = cv2.putText(frame, face_label, (x - w, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 255, 255) )
            break
        ret, jpeg = cv2.imencode(".jpg", frame)
=======
import cv2

class VideoCamera(object):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
    #for reducing frame size
    ds_factor = 0.8

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release(0)
    
    def get_frame(self):
        frame_status, frame = self.video.read()
        #resize frame
        frame = cv2.resize(frame,None, fx = self.ds_factor, fy = self.ds_factor, interpolation = cv2.INTER_AREA)
        #convert captured frame into gray-scale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #equalize image hist to increase contrast and increase feauture detection accuracy
        gray_frame = cv2.equalizeHist(gray_frame)
        #detect the face frame rectangle(top left, bottom down coordinates)
        faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        wearing_mask = None
        for (x, y, w, h) in faces:
            #increase nose detection accuracy by making the face the region of interest 
            face_frame = frame[y : y + h, x : x + w]
            resized_frame = cv2.resize(face_frame, (256, 256))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            #detect nose position
            nose = self.nose_cascade.detectMultiScale(face_frame, 1.3, 5)
    
            no_of_noses = len(nose)
            if no_of_noses == 0:
                wearing_mask = True
            else:
                wearing_mask = False
            #set frame color according to mask detected or not
            if wearing_mask == True:
                frame_color = (0, 255, 0)
            elif wearing_mask == False:
                frame_color = (0, 0, 255)
            else:
                frame_color = (255, 0, 0)
            center = (x + w//2, y + h//2)
            #frame the face
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, frame_color, 4)
            #add appropraiate label to frame
            if wearing_mask:
                face_label = "Great! You are a responsible citizen"
            
            else:
                face_label = "Go wear a mask properly!!"
                #face_label = "Don't risk your and others life.Go wear a mask!"
            frame = cv2.putText(frame, face_label, (x - w, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 255) )
            break
        ret, jpeg = cv2.imencode(".jpg", frame)
>>>>>>> afb0fbe332bc3bffd26fac8e4b8bfc2085248bb7
        return jpeg.tobytes(), wearing_mask