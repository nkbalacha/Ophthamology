import numpy as np
import os, glob
import shutil
import csv

file = "/home/nitin/Long_data/Patients.csv"
age_list =[]
eye_list=[]

with open(file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    line_count = 0
    for row in spamreader:
        print(row)
        if line_count > 0:
            age_list.append(int(row[2]))
            eye_list.append(row[4])
            eye_list.append(row[3])
        line_count +=1
        #print(', '.join(row))

new_file = "/home/nitin/Long_data/mod_Patients.csv"
#print(max(age_list))
#print(min(age_list))
count = 1
with open(new_file, mode="w") as csvfile:
    fieldnames = ['img', 'age']
    writer = csv.writer(csvfile, delimiter=',', quotechar='"')

    for age in age_list:
        writer.writerow([str(eye_list[count-1]+"_visit_1_image_1.png"),age])
        writer.writerow([str(eye_list[count-1]+"_visit_1_image_2.png"),age])
        writer.writerow([str(eye_list[count-1]+"_visit_1_image_3.png"),age])
        writer.writerow([str(eye_list[count-1]+"_visit_1_image_4.png"),age])

        writer.writerow([str(eye_list[count-1]+"_visit_2_image_1.png"),age+1])
        writer.writerow([str(eye_list[count-1]+"_visit_2_image_2.png"),age+1])
        writer.writerow([str(eye_list[count-1]+"_visit_2_image_3.png"),age+1])
        writer.writerow([str(eye_list[count-1]+"_visit_2_image_4.png"),age+1])        

        writer.writerow([str(eye_list[count]+"_visit_1_image_1.png"),age])
        writer.writerow([str(eye_list[count]+"_visit_1_image_2.png"),age])
        writer.writerow([str(eye_list[count]+"_visit_1_image_3.png"),age])
        writer.writerow([str(eye_list[count]+"_visit_1_image_4.png"),age])

        writer.writerow([str(eye_list[count]+"_visit_2_image_1.png"),age+1])
        writer.writerow([str(eye_list[count]+"_visit_2_image_2.png"),age+1])
        writer.writerow([str(eye_list[count]+"_visit_2_image_3.png"),age+1])
        writer.writerow([str(eye_list[count]+"_visit_2_image_4.png"),age+1])


        print(eye_list[count]+"_visit_2_image_4.png")
        count+=2
