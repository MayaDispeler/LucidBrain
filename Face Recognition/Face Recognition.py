#FACE RECOGNITION


#IMPORTING THE LIBRARY
import cv2

#CASCADE LOADING

Cascade_Face = cv2.CascadeClassifier('facecascade.xml')
Cascade_Eye  = cv2.CascadeClassifier('eyecascade.xml')


#FUNCTION FOR DETECTIONS

def spotted(BW, OL):
    faces = Cascade_Face.detectMultiScale(BW, 1.3, 5)
    for (x, y , z , r) in faces:
        cv2.rectangle(OL,(x, y), (x+z , y+r), (255, 0, 0), 2)
        region_BW = BW[y:y+r, x:x+z]
        region_CL = OL[y:y+r, x:x+z]
        eyes = Cascade_Eye.detectMultiScale(region_BW, 1.1, 3)
        for (e_x, e_y, e_z, e_r) in eyes:
            cv2.rectangle(region_CL,(e_x, e_y), (e_x+e_z,e_y+e_r),(0, 255, 0), 2)
    return OL

#WEBCAM TRIGGERING

Capture = cv2.VideoCapture(1)
print(Capture)

while True:
    ret, OL = Capture.read()
    print(ret,OL)
    BW = cv2.cvtColor(OL, cv2.COLOR_BGR2GRAY)
    shade = spotted(BW,OL)
    cv2.imshow('Video', shade)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
Capture.release()
cv2.destroyAllWindows()
