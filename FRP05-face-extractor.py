#!/usr/bin/env python
# coding: utf-8

from mtcnn.mtcnn import MTCNN
import cv2
import os

detector=MTCNN()

#our intention for this will be extracting faces from the celebrity dataset
os.chdir("./Brad_Pitt")
base_dir = os.getcwd()

def face_extractor(images_path,detector,face_folder,required_size=(128,128)):
    for file in os.listdir(images_path):
        for image_file in os.listdir(images_path+'\\'+file):
            file_name,file_extension = os.path.splitext(image_file)
            if (file_extension in [".png",".jpg"]):
                image = cv2.imread(images_path+'\\'+file+'\\'+image_file)
                result = detector.detect_faces(image)
                
                for i in range(len(result)):
                    (startX,startY,endX,endY)=(result[i]["box"][0], result[i]["box"][1],result[i]["box"][0]+result[i]["box"][2], result[i]["box"][1] + result[i]["box"][3])
                    confidence = result[i]["confidence"]

                    if (confidence > 0.85):
                        face_directory = base_dir+'\\'+face_folder
                        if not os.path.exists(face_directory+'\\'+file):
                            updated_face_path = face_directory+'\\'+file
                            os.makedirs(updated_face_path)
                        if (startY >= 0) and (endY >=0) and (startX >=0) and (endX >=0):
                            frame_face = image[startY:endY,startX:endX]
                            frame_face = cv2.resize(frame_face,required_size)
                            try:
                                cv2.imwrite(updated_face_path+'\\'+image_file,frame_face)
                            except:
                                pass
    return face_directory

face_extractor(celebrity_images_path,detector,"celeb_faces_images_val")

