from tkinter import *
import numpy as np
from PIL import Image
from PIL import ImageTk
import cv2, threading, os, time
from threading import Thread
from os import listdir
from os.path import isfile, join

import dlib
from imutils import face_utils, rotate_bound
import math

import pyautogui
global i
global x,y,width,height
def put_sprite(num): 
    global sprites, BTNS
    sprites[num] = (1 - sprites[num]) 
    # if sprites[num]:
    #     BTNS[num].config(relief=SUNKEN)
    # else:
    #     BTNS[num].config(relief=RAISED)

def draw_sprite(frame, sprite, x_offset, y_offset): 
    (h,w) = (sprite.shape[0], sprite.shape[1])
    (imgH,imgW) = (frame.shape[0], frame.shape[1])

    if y_offset+h >= imgH:
        sprite = sprite[0:imgH-y_offset,:,:]

    if x_offset+w >= imgW:
        sprite = sprite[:,0:imgW-x_offset,:]

    if x_offset < 0: 
        sprite = sprite[:,abs(x_offset)::,:]
        w = sprite.shape[1]
        x_offset = 0

    for c in range(3):
            frame[y_offset:y_offset+h, x_offset:x_offset+w, c] =  \
            sprite[:,:,c] * (sprite[:,:,3]/255.0) +  frame[y_offset:y_offset+h, x_offset:x_offset+w, c] * (1.0 - sprite[:,:,3]/255.0)
    return frame

def adjust_sprite2head(sprite, head_width, head_ypos, ontop = True):
    (h_sprite,w_sprite) = (sprite.shape[0], sprite.shape[1])
    factor = 1.0*head_width/w_sprite
    sprite = cv2.resize(sprite, (0,0), fx=factor, fy=factor)
    (h_sprite,w_sprite) = (sprite.shape[0], sprite.shape[1])

    y_orig =  head_ypos-h_sprite if ontop else head_ypos 
    if (y_orig < 0):
            sprite = sprite[abs(y_orig)::,:,:] 
            y_orig = 0
    return (sprite, y_orig)


def apply_sprite(image, path2sprite,w,x,y, angle, ontop = True):
    sprite = cv2.imread(path2sprite,-1)
    sprite = rotate_bound(sprite, angle)
    (sprite, y_final) = adjust_sprite2head(sprite, w, y, ontop)
    image = draw_sprite(image,sprite,x, y_final)

def calculate_inclination(point1, point2):
    x1,x2,y1,y2 = point1[0], point2[0], point1[1], point2[1]
    incl = 180/math.pi*math.atan((float(y2-y1))/(x2-x1))
    return incl


def calculate_boundbox(list_coordinates):
    x = min(list_coordinates[:,0])
    y = min(list_coordinates[:,1])
    w = max(list_coordinates[:,0]) - x
    h = max(list_coordinates[:,1]) - y
    return (x,y,w,h)

def detectUpperBody(image):
    cascadePath = "data/haarcascade_upperbody.xml"
    result = image.copy()
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascadePath)
    Rect = cascade.detectMultiScale(imageGray, scaleFactor=1.1, minNeighbors=1, minSize=(1,1)) 
    if len(Rect) <= 0:
	    return False	
    else:
	    return Rect

def get_face_boundbox(points, face_part):
    if face_part == 1: # left eye brow
        (x,y,w,h) = calculate_boundbox(points[17:22])
    elif face_part == 2:# right eye brow
        (x,y,w,h) = calculate_boundbox(points[22:27]) 
    elif face_part == 3: # left eye
        (x,y,w,h) = calculate_boundbox(points[36:42])
    elif face_part == 4: # right eye
        (x,y,w,h) = calculate_boundbox(points[42:48])
    elif face_part == 5: # nose
        (x,y,w,h) = calculate_boundbox(points[29:36])
    elif face_part == 6: #jawline
        (x,y,w,h) = calculate_boundbox(points[0:17])
    elif face_part == 7: 
        # (x,y,w,h) = calculate_boundbox(points[48:68]) #mouth
        (x,y,w,h) = calculate_boundbox(points[1:5])
    elif face_part == 8:
        (x,y,w,h) = calculate_boundbox(points[12:16])
    return (x,y,w,h)

image_path = ''

def add_sprite(img):
    global image_path
    image_path = img
    # print(img.rsplit('/',1))
    put_sprite(int(img.rsplit('/',1)[0][-1])) #getting image ka naam , and calling putsprite


def screenshot(i):
    # take screenshot using pyautogui 
    image = pyautogui.screenshot(region=(1630,800, 620, 500)) 
   
# since the pyautogui takes as a PIL(pillow) and in RGB we need to convert it to numpy array and BGR so we can write it to the disk 
    image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR) 
   
# writing it to the disk using opencv 
    #cv2.imwrite("image%no.png" %i, image)
    
    cv2.imwrite('static/images/compare/'+sys.argv[1].replace('/','-') +  '.jpg', image)
    i = i + 1
    
#Principal Loop where openCV (magic) ocurs #jadoo
def cvloop(run_event):
    global panelA
    global sprites
    global image_path
    i = 0
    video_capture = cv2.VideoCapture(0) #read from webcam
    (x,y,w,h) = (0,0,10,10) #whatever initial values

    #Filters path
    detector = dlib.get_frontal_face_detector()

    model = "data/shape_predictor_68_face_landmarks.dat" # we are getting facial landmarks based on the model
    predictor = dlib.shape_predictor(model) # link to model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

    while run_event.is_set(): 
        ret, image = video_capture.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        faces = detector(gray, 0)

        for face in faces: 
            (x,y,w,h) = (face.left(), face.top(), face.width(), face.height()) #x,y,w,h

            shape = predictor(gray, face) 
            shape = face_utils.shape_to_np(shape) # convertig face into array
            incl = calculate_inclination(shape[17], shape[26]) # inclination based on eyebrows # kuch kuch maths

            # condition to see if mouth is open
            is_mouth_open = (shape[66][1] -shape[62][1]) >= 10 # if distance between lips is greater than or equal to 10 we take mouth is open

            if sprites[0]:
                apply_sprite(image,image_path,w,x,y+40, incl, ontop = True) 
        

            if sprites[1]:
                (x1,y1,w1,h1) = get_face_boundbox(shape, 6)
                apply_sprite(image,image_path,w1,x1,y1+280, incl)

                #Tiara
            if sprites[3]:
                (x3,y3,_,h3) = get_face_boundbox(shape, 1)
                apply_sprite(image,image_path,w,x,y3, incl, ontop = True)

    
            (x0,y0,w0,h0) = get_face_boundbox(shape, 6) #bound box of mouth
            #earrings
            if sprites[2]:
                (x3,y3,w3,h3) = get_face_boundbox(shape, 7) #nose
                apply_sprite(image, image_path,w3,x3-14,y3+50, incl)
                (x3,y3,w3,h3) = get_face_boundbox(shape, 8) #nose
                apply_sprite(image, image_path,w3,x3+14,y3+50, incl)
            #tops
            if sprites[4]:
                (x3,y3,w3,h3) = get_face_boundbox(shape, 7) #nose
                apply_sprite(image, image_path,w3,x3-20,y3+25, incl)
                (x3,y3,w3,h3) = get_face_boundbox(shape, 8) #nose
                apply_sprite(image, image_path,w3,x3+20,y3+25, incl)
            # frocks
            if sprites[5]:
                (x3,y3,_,h3) = get_face_boundbox(shape, 1)
                apply_sprite(image,image_path,w+600,x-320,y3+100, incl, ontop = False)

                
                

            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        panelA.configure(image=image)
        panelA.image = image

    video_capture.release()

# Initialize GUI object
def center_window(width=400, height=400):
    # get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # calculate position x and y coordinates
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))


root = Tk()
center_window(650, 700)
root.title("E-Dressing- Face")
this_dir = os.path.dirname(os.path.realpath(__file__))

btn1 = None
btn2 = None

def try_on(image_path):
    i=0
    btn1 = Button(root, text="Try it ON", command = lambda:add_sprite(image_path))
    btn1.pack(side="top", fill="both", expand="no", padx="5", pady="5")
#new line
    
    btn2 = Button(root, text="ScreenShot", command = lambda:screenshot(i))
    i = i + 1
    btn2.pack(side="top", fill="both", expand="no", padx="5", pady="5")
panelA = Label(root)
panelA.pack( padx=10, pady=10)

sprites = [0,0,0,0,0,0]
BTNS = [btn1, btn2]

try_on(sys.argv[1])
run_event = threading.Event()
run_event.set()
action = Thread(target=cvloop, args=(run_event,))
action.setDaemon(True)
action.start()

def terminate():
        global root, run_event, action
        run_event.clear()
        time.sleep(1)
        root.destroy()

root.protocol("WM_DELETE_WINDOW", terminate)

root.mainloop() 
