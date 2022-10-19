import numpy as np
import os, glob
import shutil

path = "/home/nitin/Long_data/FundusImagesColor"
path_new = "/home/nitin/Long_data/mod_FundusImages"


for i in range(1,141):
    loc=""
    if i < 10: 
        loc = str(0)+str(0) +str(i)
    if ((i < 100) and (i >=10)): 
        loc =str(0) +str(i)
    if (i>=100):
        loc = str(i)

    path_img = path+"/eye_"+ loc
    dir_list = os.listdir(path_img)
    for j in dir_list:
        print(j)
        shutil.copy(path_img +"/"+j, path_new +"/eye_"+loc+"_"+j)
