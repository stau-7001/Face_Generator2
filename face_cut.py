import argparse
import json
import os
from unicodedata import name

import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import string

import face_recognition
import numpy as np
from nbformat import read
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from align_face_FFHQ import align_face
from id_generator import get_id
from insightface_func.face_detect_crop_single import *

Max_int = 40
Min_int = 15

def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--res', type=int, default=512,
                                            help="expected image resolution")
    parser.add_argument('-t', '--tol', type=int, default=500,
                                            help="tolerance range for the size of the captured face image")
    parser.add_argument('-m', '--mode', type=int, default=0,
                                            help="0:VGG crop, 1:FFHQ crop")
    parser.add_argument('-i', '--interval', type=int, default=5,
                                            help="number of frames interval")
    parser.add_argument('-s', '--skip', type=int, default=60,
                                            help="number of frames to be skipped")
    parser.add_argument('-b', '--blur', type=int, default=100,
                                            help="threshold value for out-of-focus blur detection")
    return parser.parse_args()

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
 
def res_checker(face_locations = None, res=512, tol = 400):
    top, right, bottom, left = face_locations[0]
    if abs(top-bottom)-abs(right-left) in range(-50, 51):
        # print(face_locations[0])
        if abs(top-bottom) > res-tol and abs(right-left) > res-tol:
            return 1
    return 0
    

def frame_generator_VGG(Vid,filename, opt = None):
    cap = cv2.VideoCapture('./video/'+filename)
    with open("./dataset/"+str(Vid).zfill(4)+"/id_record.json","a") as f:
        f.write('[')
    if opt.mode != 1:
        app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=None)

    frame_id = -1
    discard = 0
    interval = 0
    ss = 0
    gray_lap = 0
    flag = False
    pre_img = None
    while cap.isOpened():
        frame_id = frame_id +1
        ret, frame = cap.read()
        if discard > 0 or interval > 0:
            discard = discard - 1
            interval = interval -1
            continue
        if ret == True:
            # cv2.imwrite("temp.png", frame)
            # raw_image = face_recognition.load_image_file("temp.png")
            same_flag = False
            face_locations = face_recognition.face_locations(frame)
            if app.get(frame,opt.res) != None and len(face_locations) > 0 and res_checker(face_locations, opt.res, opt.tol):
                cv_images, _ = app.get(frame,opt.res)
                pil_image = Image.fromarray(cv2.cvtColor(cv_images[0],cv2.COLOR_BGR2RGB))
                img_array = np.asarray(pil_image)

                if np.all(pre_img != None):
                        ss=ssim(pre_img,img_array,multichannel=True)
                        # print(ss)
                        if ss > 0.6:
                            opt.interval = min(opt.interval**2, Max_int)
                            if ss > 0.7:
                                same_flag = True
                        else:
                            opt.interval = Min_int
                interval = opt.interval
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                gray_lap = cv2.Laplacian(img_gray,cv2.CV_64F).var()
                if gray_lap >= opt.blur and not same_flag:
                    pre_img = img_array
                    img_id = get_id(pil_image)
                    f_path ="./dataset/"+str(Vid).zfill(4)+"/"+str(Vid).zfill(4)+"_"+str(frame_id).zfill(6)+"_"+str(gray_lap)+".png"
                    pil_image.save(fp=f_path)
                    write_dict = {f_path:img_id}
                    with open("./dataset/"+str(Vid).zfill(4)+"/id_record.json","a") as f:
                        if flag: 
                            f.write(',')
                        json.dump(write_dict,f,indent=1)
                        flag = True            
            else:
                discard = opt.skip
        else:
            break
    with open("./dataset/"+str(Vid).zfill(4)+"/id_record.json","a") as f:
        f.write(']')
    cap.release()

def frame_generator_FFHQ(Vid,filename, opt = None):
    with open("./dataset/"+str(Vid).zfill(4)+"/id_record.json","a") as f:
        f.write('[')
    cap = cv2.VideoCapture('./video/'+filename)

    frame_id = -1
    discard = 0
    interval = 0
    ss = 0
    gray_lap = 0
    flag = False
    pre_img = None
    while cap.isOpened():
        frame_id = frame_id +1
        ret, frame = cap.read()
        if discard > 0 or interval > 0:
            discard = discard - 1
            interval = interval -1
            continue
        if ret==True:
            # cv2.imwrite("temp.png", frame)
            # raw_image = face_recognition.load_image_file("temp.png")
            same_flag = False
            face_locations = face_recognition.face_locations(frame)
            pil_image = align_face(frame, opt.res)
            if pil_image != None and len(face_locations) > 0 and res_checker(face_locations, opt.res, opt.tol):
                img_array = np.asarray(pil_image)
                if np.all(pre_img != None):
                        ss=ssim(pre_img,img_array,multichannel=True)
                        # print(ss)
                        if ss > 0.6:
                            opt.interval = min(opt.interval**2, Max_int)
                            if ss > 0.7:
                                same_flag = True
                        else:
                            opt.interval = Min_int
                interval = opt.interval
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                gray_lap = cv2.Laplacian(img_gray,cv2.CV_64F).var()
                if gray_lap >= opt.blur and not same_flag:
                    pre_img = img_array
                    img_id = get_id(pil_image)
                    f_path ="./dataset/"+str(Vid).zfill(4)+"/"+str(Vid).zfill(4)+"_"+str(frame_id).zfill(6)+".png"
                    pil_image.save(fp=f_path)
                    write_dict = {f_path:img_id}
                    with open("./dataset/"+str(Vid).zfill(4)+"/id_record.json","a") as f:
                        if flag: 
                            f.write(',')
                        json.dump(write_dict,f,indent=1)
                        flag = True            
            else:
                discard = opt.skip
    with open("./dataset/"+str(Vid).zfill(4)+"/id_record.json","a") as f:
        f.write(']')
    cap.release()
   
        
if __name__ == '__main__':
    config = getParameters()
    test_list=[]
    test_list = os.listdir("./video")
    print(test_list)
    # with open("List_of_testing_videos.txt", "r") as f:
    #     test_list= f.readlines()
    id = 0
    li=len(test_list)
    for i in tqdm(range(0,li)):
        print("\n Starting processing video", test_list[i])
        # print(i)
        path = "./dataset/"+str(id).zfill(4)
        if not os.path.exists(path):
            os.makedirs(path)
        if config.mode == 0:
            frame_generator_VGG(id, test_list[i], config)
        elif config.mode == 1:
            frame_generator_FFHQ(id, test_list[i], config)
        id=id+1
    #print(test_list)
        