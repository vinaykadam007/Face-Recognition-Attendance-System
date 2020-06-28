import tkinter as tk
from tkinter import Message ,Text
import os
# import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
# import tkinter.ttk as ttk
import tkinter.font as font

import cv2




################################################################################################
#################################  GUI  Code PART 1 ############################################
################################################################################################

## GUI WINDOW
window = tk.Tk()
window.title("FRAS")
 
#window configuration
# window.geometry('1280x720')
window.configure(background='black')
window.attributes('-fullscreen', True)
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

#Title of App
message = tk.Label(window, text="Face Recognition Based Attendance Management System"  ,fg="white", bg="green"  ,height=3,font=('times', 30, 'bold')) 
message.place(x=200, y=10)

#Enter ID field
lbl = tk.Label(window, text="Enter ID : ",width=20  ,height=2  ,fg="white"  ,bg="black" ,font=('times', 20, ' bold ') ) 
lbl.place(x=200, y=200)
txt = tk.Entry(window,width=30  ,bg="white" ,fg="red",font=('times', 15, ' bold '))
txt.place(x=500, y=220)

#Enter Name field
lbl2 = tk.Label(window, text="Enter Name : ",width=20  ,fg="white"  ,bg="black"    ,height=2 ,font=('times', 20, ' bold ')) 
lbl2.place(x=200, y=300)
txt2 = tk.Entry(window,width=30  ,bg="white"  ,fg="red",font=('times', 15, ' bold ')  )
txt2.place(x=500, y=320)

#Notification
lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="red"  ,bg="yellow"  ,height=2 ,font=('times', 15, ' bold')) 
lbl3.place(x=200, y=400)
#notification Message
message = tk.Label(window, text="" ,bg="white"  ,fg="red"  ,width=50  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message.place(x=500, y=400)

#Attendance Notification (Implement it in scrolled text later)
lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="red"  ,bg="yellow"  ,height=2 ,font=('times', 15, ' bold')) 
lbl3.place(x=200, y=600)
message2 = tk.Label(window, text="" ,fg="red"   ,bg="white",activeforeground = "green",width=50  ,height=2  ,font=('times', 15, ' bold ')) 
message2.place(x=500, y=600)
 
#Functions to update or clear Notification Item in GUI
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)
def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
#Functions to check whether given ID is number or not
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 




################################################################################################
#################################  Main Functions ##############################################
################################################################################################



##################################
########## TAKE IMAGE ############
##################################

# This function is used to take  call TAKE IMAGES and store in images  
def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    #if the entered id is numeric then execute
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows()

        ## saved images id and his name  >> format >> " name.id.pic_num.jpg "
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        ## Write the student details i.e add a new row in STUDENTDETAILS.CSV append+ mode
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res) #show notification 
    else:
        #check is that nmber is float or not if float show msg
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        # check is that number is string or not if string show this msg
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    



##################################
########## TRAIN MODEL ###########
##################################
# Train Model with new images and save the model
def TrainImages():
    # recognizer = cv2.face_FisherFaceRecognizer.create()
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)

# to get all images and it's ids in 2 array as x->contain image path  y-> contain image id(taken from image name)
def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids








##################################
########## TRACK IMAGES ##########
##################################
# It will take new image and mark attendance for that student
def TrackImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    # recognizer = cv2.face_FisherFaceRecognizer.create()
    # recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    print(df)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    #Create an attendance sheet
    col_names =  ['Id','Name','Date','Time']
    attendance_sheet = pd.DataFrame(columns = col_names)   
    
    taken_attendance ={
        'id':'',
        'name':'',
        'date':'',
        'timestamp':''
    }

    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)  
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w]) 
            # print(conf)                                  
            if(conf < 50):
                #get current timestamp
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                Name=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+Name
                # attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                # attendance.loc[len(attendance)] = oneStuAttt[0]
                # print(oneStuAttt)
                
                # oneStuAttt.append([Id,aa,date,timeStamp])
                taken_attendance['id'] = str(Id)
                taken_attendance['name'] = str(Name)
                taken_attendance['date'] = str(date)
                taken_attendance['timestamp'] = str(timeStamp)
                print(taken_attendance)
            else: 
                taken_attendance['id'] = 'Unknown'
                taken_attendance['name'] = ''
            
            # if(conf > 75):
            #     noOfFile=len(os.listdir("ImagesUnknown"))+1
            #     # cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w]) 
            print(type(taken_attendance.get("id")))     
            square_name = taken_attendance.get("id") + ' - ' + taken_attendance.get("name")
            cv2.putText(im,str(square_name),(x,y+h), font, 1,(255,255,255),2)    

        # attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
        
        


    # attendance.loc[len(attendance)] = oneStuAttt[0]
    #create a new csv file and save attendace file
    attendance_sheet.loc[len(attendance_sheet)] = [taken_attendance.get("id"),taken_attendance.get("name"),taken_attendance.get("date"),taken_attendance.get("timestamp")]
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance_sheet.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res='Taken Attendance for : '+ square_name +'\n saved attendance in excel'
    message2.configure(text= res)










################################################################################################
#################################  GUI  Code PART 2 ############################################
################################################################################################
  
clearButton = tk.Button(window, text="Clear", command=clear  ,fg="black"  ,bg="orange"  ,width=10  ,height=1 ,activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=850, y=210)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="black"  ,bg="orange"  ,width=10  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton2.place(x=850, y=310)  

takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="white"  ,bg="blue"  ,width=10  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)

trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="white"  ,bg="blue"  ,width=10  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=367, y=500)

trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="white"  ,bg="blue"  ,width=10  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
trackImg.place(x=534, y=500)

quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="black"  ,bg="red" ,width=10  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=701, y=500)

 
window.mainloop()