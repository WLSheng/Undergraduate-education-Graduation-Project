import cv2
import RPi.GPIO as GPIO
import time,sys
import numpy as np
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
    
#left is b34,right is a12

ENA = 5
IN1 = 6
IN2 = 13
IN3 = 21
IN4 = 20
ENB = 16
    
    
GPIO.setup(ENA,GPIO.OUT)
GPIO.setup(IN1,GPIO.OUT)
GPIO.setup(IN2,GPIO.OUT)
GPIO.setup(IN3,GPIO.OUT)
GPIO.setup(IN4,GPIO.OUT)
GPIO.setup(ENB,GPIO.OUT)
  
pwm_a = GPIO.PWM(ENA , 50)
pwm_a.start(100)
pwm_b = GPIO.PWM(ENB , 50)
pwm_b.start(100)
    
GPIO.output(IN1,GPIO.HIGH)
GPIO.output(IN2,GPIO.LOW)
GPIO.output(IN3,GPIO.HIGH)
GPIO.output(IN4,GPIO.LOW)


def stop():
    GPIO.output(IN1,GPIO.LOW)
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,GPIO.LOW)
    GPIO.output(IN4,GPIO.LOW)
    
def right(speed,w):
    width = int(w / 2)
    new = width-(speed/2 - width/2)
    if(new>=100):
        new = 100
    print("right speed change to  :",new)
    pwm_a.ChangeDutyCycle(new) #geng gai you lun de sudu
    pwm_b.ChangeDutyCycle(100)


def left(speed,w):
    width = int(w / 2)
    new = width-(speed/2 - width/2)
    if(new>=100):
        new = 100
    print("left speed change to :",new)
    pwm_b.ChangeDutyCycle(new)
    pwm_a.ChangeDutyCycle(100)


def plan(frame,width,x11,x12,x21,x22):
    
    jude = 0
    for i in range(width-1,int(width/2),-1):
        if frame.item(i, x11) >= 200 and frame.item(i, x12)>= 200 :
            jude = 1
            for j in range(width-1,int(width/2),-1):
                if frame.item(j, x21) >= 200 or frame.item(j, x22) >= 200 :
                    if frame.item(j,x21) >= 200 and frame.item(j,x22) >= 200:
                        return(3,j)  
                    elif frame.item(j, x21) >=  200:
                        #print "two1 is :",j
                        return (1,j)
                    elif frame.item(j, x22) >= 200 :
                        #print "two2 is: ", j
                        return (2,j)
                    
        if frame.item(i, x11) >= 200 or frame.item(i, x12) >= 200 :
            jude = 1
            if frame.item(i, x11) >= 200 :
                #print "one1 is : ",i
                return (1,i)
            elif frame.item(i, x12) >= 200 :
                #print "one2 is : ",i
                return (2,i)
    if(jude == 0):
        return (0,width/2)


def canny(img):
    #截取下半部分的车道线
    height, width, _ = img.shape

    gass = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(gass, 50, 200, 3)
    resultt = np.zeros((height, width), np.uint8)
    result = np.zeros((height, width), np.uint8)
    resultt = cv2.rectangle(resultt, (0, int(height/1.7)), (width, height), (255), thickness=cv2.FILLED)  # 填充的矩形,这里调近视景区的参数
    lines = cv2.HoughLines(canny, 1, np.pi / 180, 118)  # 这里对最后一个参数使用了经验型的值
    i = 0
    while i < 2:
        for line in lines[i]:
            i = i + 1
            rho = line[0]  # 第一个元素是距离rho
            theta = line[1]  # 第二个元素是角度theta
            # print(rho, theta)
            if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                # 该直线与第一行的交点
                pt1 = (int(rho / np.cos(theta)), 0)
                # 该直线与最后一行的焦点
                pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                # 绘制一条白线
                cv2.line(result, pt1, pt2, 255, 10)
            else:  # 水平直线
                # 该直线与第一列的交点
                pt1 = (0, int(rho / np.sin(theta)))
                # 该直线与最后一列的交点
                pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                # 绘制一条直线
                cv2.line(result, pt1, pt2, 255, 10)
    r = cv2.bitwise_and(result, resultt)
    return r


def white_line(img):
    kernel = np.ones((5, 5), np.uint8)
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    threshold = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 新方法HSV的
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([100, 35, 255])
    Rthresh = cv2.inRange(hsv, lower_red, upper_red)
    Rthreshold = cv2.morphologyEx(Rthresh, cv2.MORPH_OPEN, kernel)
    new_threshold = cv2.bitwise_and(Rthreshold, threshold)
    new_threshold = cv2.morphologyEx(new_threshold, cv2.MORPH_CLOSE, kernel)

    # 矩形与车道线的并运算
    rect = np.zeros((height, width), np.uint8)
    rect = cv2.rectangle(rect, (0, int(height/2.5)), (width, height), (255), thickness=cv2.FILLED)  # 填充的矩形,这里调近视景区的参数
    r = cv2.bitwise_and(new_threshold, rect)
    # 画在原图上的车道线
    # image, contours, hierarchy = cv2.findContours(r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # new_cnt = []
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area > 300:
    #         # print(area)
    #         new_cnt.append(cnt)
    # img = cv2.drawContours(img, new_cnt, -1, (0, 0, 255), thickness = cv2.FILLED)
    return r


if __name__=="__main__":
    video_time = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
    
    video_name = str(video_time) + '.avi'
    print(video_name)
    cap = cv2.VideoCapture(0)  
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_name,fourcc,25,(640, 480))
    fps = 1
    t2 = 0
    t3 = 0
    cv2.namedWindow("black",0)
    cv2.namedWindow("frame",0)
    #count_fps = cap.get(7)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())        
    car_cascade = cv2.CascadeClassifier('cars.xml')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    
    while (cap.isOpened()):
        gg,frameY = cap.read()#will return two value
        fps +=1
        if gg == True:
            out.write(frameY)
            if(fps % 13 ==0):
                t1 = cv2.getTickCount()
                frame = cv2.resize(frameY,(200,200))
                #jin zi ta

                #frame = cv2.pyrDown(frameY)
                #frame = cv2.pyrDown(frame)
                
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                #road
                thresh = canny(gray)
                # thresh = white_line(frame)

                black = gray.copy()
                black[:] =  [0]
                #print gray.shape,black.shape
                #people
                (rects,weights) = hog.detectMultiScale(gray,winStride=(4,4),padding=(8,8),scale = 1.05)
                for(x,y,w,h) in rects:
                    cv2.rectangle(frame,(x+20,y+10),(x+w-0,y+h-10),(0,255,0),1)
                    cv2.rectangle(black,(x,y),(x+w,y+h),(255),1)
                #   car
                cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, flags=8, minSize=None, maxSize=None)
                for(xx,yy,ww,hh) in cars:
                    cv2.rectangle(frame,(xx,yy),(xx+ww,yy+hh),(255,0,0),1)
                    cv2.rectangle(black,(xx,yy),(xx+ww,yy+hh),(255),1)
                    #cars = cars + 1
                #end
                thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
                black = cv2.bitwise_or(black,thresh)
                black1 = black.copy()
                
                x11 = 90
                x12 = 105
                x21 = 30
                x22 = 160
                width = 200
                for x in range(0,width-1,1):
                    black1[x,x11] = 255
                    black1[x,x12] = 255
                    black1[x,x21] = 255
                    black1[x,x22] = 255
                
                cv2.imshow("frame",frame)
                cv2.imshow("black",black1)
                #cv2.imshow("gray", gray)
                jude, speed = plan(black,width,x11,x12,x21,x22)#jude = 012,1 turn to light,2 left
                if(jude == 3):
                    pwm_b.ChangeDutyCycle(50)          
                    pwm_a.ChangeDutyCycle(50)
                    print("stop!!!")
                elif(jude == 1):
                    right(speed,width)
                elif(jude == 2):
                    left(speed,width)
                elif(jude == 0):
                    pwm_b.ChangeDutyCycle(100)          
                    pwm_a.ChangeDutyCycle(100)
                t2 = cv2.getTickCount()
                t3 = (t2 - t1)/cv2.getTickFrequency()
                
                print("this frame :",fps,"spend time is:",t3)
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    stop()
                    break
        else:
            stop()
            break
    stop()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()

#i = cap.get(5)#()write number
