import cv2
import numpy as np
import face_recognition

# loading the images
imgMahima = face_recognition.load_image_file('ImageBasic/Mahima Bhatt.jpeg')
imgMahima = cv2.cvtColor(imgMahima,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/Unnati Bhatt.jpeg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgMahima)[0]
encodeMahima = face_recognition.face_encodings(imgMahima)[0]
cv2.rectangle(imgMahima,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#print(faceLoc)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeMahima],encodeTest)
faceDis = face_recognition.face_distance([encodeMahima],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_ITALIC,1,(0,0,255),2)


cv2.imshow('Mahima Bhatt',imgMahima)
cv2.imshow('Mahima Test',imgTest)
cv2.waitKey(0)