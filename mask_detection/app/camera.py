import cv2

class VideoCamera(object):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
    mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
    #for reducing frame size
    ds_factor = 1.29

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()
    
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
            half_face = frame[y + h//2 + h//5 : y + h, x : x + w]
            resized_frame = cv2.resize(face_frame, (256, 256))
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            #detect nose position
            nose = self.nose_cascade.detectMultiScale(face_frame, 1.3, 5)
            mouth = self.mouth_cascade.detectMultiScale(half_face, 1.3, 5)
            no_of_noses = len(nose)
            no_of_mouth = len(mouth)

            if no_of_noses == 0 and no_of_mouth == 0:
                wearing_mask = True
                frame_color = (0, 255, 0)
                face_label = "Thank you for following the safety guidelines"
                X = x -w -w//11
            elif no_of_mouth > 0:
                wearing_mask = False
                frame_color = (0, 0, 255)
                face_label = "Please follow the safety guidelines and wear a mask"
                X = x -w -w//11
            else:
                wearing_mask = False
                frame_color = (0, 0, 255)
                face_label = "Please ensure you are wearing a mask properly"
                X = x -w -w//3

            center = (x + w//2, y + h//2)
            #frame the face
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, frame_color, 4)
            #frame = cv2.rectangle(frame, (x,y +h//2 + h//5), (x + w, y + h), (0,0,0), 3)
            #frame mouth
            # for (xx,yy,ww,hh) in mouth:
            #     cv2.rectangle(frame, (x +xx,y + yy + h//2 + h//5), (xx + x +ww,y + yy+hh + h//2 + h//5), (255,0,0), 3)
            #     break
            #frame mouth
            # for (xx,yy,ww,hh) in nose:
            #     cv2.rectangle(frame, (x +xx,y + yy ), (xx + x +ww,y + yy+hh ), (255,0,0), 3)
            #     break
           
            cv2.rectangle(frame, (x - 3*w, y + h +h//3), (x + 5*w, y + 2*h + h//3), (0,0,0), -1)
            frame = cv2.putText(frame, face_label, (X , y + h + h//2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255) )
            break
        ret, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes(), wearing_mask