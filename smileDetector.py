
import cv2
from cv2 import COLOR_BGR2GRAY

# Pre-Trained Face and Smile classifiers:
trained_face_detector=cv2.CascadeClassifier('haarcascade_frontalfacedetector.xml')
trained_smile_detector=cv2.CascadeClassifier('haarcascade_smiledetector.xml')

# Taking webcam fotage 
webcam=cv2.VideoCapture(0)

# Show the current frame:
while True:
        successful_read,frame=webcam.read()
        # For Safe coding (checking if frame is read successfully or not)
        if not successful_read:
            break
        
        # changing frame to black and white:
        grayscalledFrame=cv2.cvtColor(frame,COLOR_BGR2GRAY)
        
        # Getting co-ordinates of face:
        faceCoordinates=trained_face_detector.detectMultiScale(grayscalledFrame)
        
        #looping through all the faces detected above:
        for (x,y,w,h) in faceCoordinates:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,200,50),5)
            
            #Get the sub frame(using numpy N-dimensional array):
            the_face=frame[y:y+h,x:x+w] 
            
            face_grayscale=cv2.cvtColor(the_face,COLOR_BGR2GRAY)
            
            smileCoordinates=trained_smile_detector.detectMultiScale(face_grayscale,scaleFactor=1.5,minNeighbors=20)
            #  finding a smile within all FACES:
            # for (x_,y_,w_,h_) in smileCoordinates:
            #     cv2.rectangle(the_face,(x_,y_),(x_+w_,y_+h_),(20,20,200),2)
            if len(smileCoordinates) >0:
                cv2.putText(frame,"SMILING",(x,y+h+40),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))

            
        ##Show current frame:
        cv2.imshow("Gaurav Bora's Smile Detctor",frame)

        key=cv2.waitKey(1)
        if key== 81 or key==113:
            print('program quitted')
            break
webcam.release()
cv2.destroyAllWindows()