import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
import pickle
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json
import time
import joblib
import sys

application = Flask(__name__)
api = Api(application)

# model1 = pickle.load(open('model.pkl', 'rb'))
# model2 = pickle.load(open('model1.pkl', 'rb'))




class spoof_detection(Resource):
    def get(self):
        '''
        For rendering results on HTML GUI
        '''
        root_dir = os.getcwd()
        # Load Face Detection Model
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        # Load Anti-Spoofing Model graph
        json_file = open('finalyearproject_antispoofing_model_mobilenet.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load antispoofing model weights 
        model.load_weights('finalyearproject_antispoofing_model_99-0.976842.h5')
        print("Model loaded from disk")
        # video.open("http://192.168.1.101:8080/video")
        # vs = VideoStream(src=0).start()
        # time.sleep(2.0)

        video = cv2.VideoCapture(0)
        while True:
            # start = time.time() + 60 # CHANGE start time counter
            # delta = 60
            # print(start - time.time())
            try:
                ret,frame = video.read()
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.3,5)
                for (x,y,w,h) in faces:  
                    face = frame[y-5:y+h+5,x-5:x+w+5]
                    resized_face = cv2.resize(face,(160,160))
                    resized_face = resized_face.astype("float") / 255.0
                    # resized_face = img_to_array(resized_face)
                    resized_face = np.expand_dims(resized_face, axis=0)
                    # pass the face ROI through the trained liveness detector
                    # model to determine if the face is "real" or "fake"
                    preds = model.predict(resized_face)[0]
                    
                    # print(preds)
                    if preds > 0.30:
                        label = 'spoof'
                        cv2.putText(frame, label, (x,y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        cv2.rectangle(frame, (x, y), (x+w,y+h),
                            (0, 0, 255), 2)
                        def message(preds):
                            if preds >= 0.99:
                                print("SPOOFED FACE DETECTED")
                        message(preds)
                        break

                
                        
                    if preds < 0.001:
                        label = 'real'
                        cv2.putText(frame, label, (x,y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        cv2.rectangle(frame, (x, y), (x+w,y+h),
                        (0, 255, 0), 2)
                    
                    
                    

                cv2.imshow('frame', frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            except Exception as e:
                pass
        video.release()        
        cv2.destroyAllWindows()

api.add_resource(spoof_detection , '/Detection')

if __name__ == "__main__":
    application.run(debug=True)