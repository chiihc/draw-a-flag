import numpy as np
import cv2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output455.avi', fourcc, 60.0, (1620, 1080),True)

red=(np.array([np.zeros(1749600),np.zeros(1749600),np.ones(1749600)*255],dtype='uint8').T).reshape((1080,1620,3))


def inshow_and_write(frame):
    cv2.imshow('frame',frame)
    out.write(frame)
    
def draw_circle(x_mid,y_mid,r,color):
    global frame
    for i in range(r):
        for x in range(x_mid-i,x_mid+i+1):
            for y in range(y_mid-i,y_mid+i+1):
                if (x-x_mid)*(x-x_mid)+(y-y_mid)*(y-y_mid)<i*i:
                    frame[x,y]=[0,255,255]
        inshow_and_write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def draw_arc(x_mid,y_mid,r,color):
    global frame
    for i in np.arange(0,np.pi/2,0.01):
        for x in range(x_mid-r,x_mid+r+1):
            for y in range(y_mid-r,y_mid+r+1):
                if (x-x_mid)*(x-x_mid)+(y-y_mid)*(y-y_mid)<=r*r:
                    if (x-x_mid)*(x-x_mid)+(y-y_mid)*(y-y_mid)>=(r-1)*(r-1):
                        if y<x*np.tan(i):
                            frame[x,y]=color
        inshow_and_write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def draw_line(x1,x2,y1,y2):
    for x in range(min(x1,x2),max(x1,x2)+1):
        for y in range(min(y1,y2),max(y1,y2)+1):
            if np.abs((y1-y2)*(x-x2)+(x1-x2)*(y2-y))<=np.sqrt(((y1-y2)*(y1-y2)+(x1-x2)*(x1-x2))):
                frame[x,y]=[0,255,255]
                inshow_and_write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

def fill_color_five_star(x_mid,y_mid,r,theta):
    x1=np.cos(0.4*np.pi*np.array([0,1,2,3,4,0,1,2,3,4])+np.pi)*r
    y1=np.sin(0.4*np.pi*np.array([0,1,2,3,4,0,1,2,3,4])+np.pi)*r
    x1,y1=np.rint(np.dot(theta,np.array([x1,y1])))
    x1+=x_mid
    y1+=y_mid
    def f_distance(x1,x2,y1,y2,x,y):
        return (y1-y2)*(x-x2)+(x1-x2)*(y2-y)
    for i in range(256):
        for x in range(x_mid-r,x_mid+r+1):
            for y in range(y_mid-r,y_mid+r+1):
                i2=0
                for i1 in range(5):
                    if f_distance(x1[i1],x1[i1+2],y1[i1],y1[i1+2],x,y)>=0:
                        i2+=1
                        if i2>=4:
                            break
                if i2>=4:
                    frame[x,y]=[0,i,255]
                    true_flag[x,y]=[0,i,255]
        inshow_and_write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

for i in range(1):
    frame=red
    inshow_and_write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
true_flag=frame

#画横线
for i in range(162):
    frame[538:541,i*10:i*10+10]=np.array([0,0,0])
    inshow_and_write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#画竖线
for i in range(108):
    frame[i*10:i*10+10,808:811]=np.array([0,0,0])
    inshow_and_write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#画横线
for i1 in range(9):
    for i in range(81):
        frame[54+54*i1,i*10:i*10+10]=np.array([0,0,0])
        inshow_and_write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#画竖线
for i1 in range(14):
    for i in range(54):
        frame[i*10:i*10+10,54+54*i1]=np.array([0,0,0])
        inshow_and_write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 画圆
x_mid=270
y_mid=270
r=5
draw_circle(x_mid,y_mid,r,[0,i,255])

# 画圆弧
x_mid=270
y_mid=270
r=54*3
draw_arc(x_mid,y_mid,r,[0,0,0])

# 画圆
x_mid_center=270
y_mid_center=270
r=5
for i1 in range(5):
    x_mid=x_mid_center+int(np.cos(0.4*np.pi*i1+np.pi)*54*3)
    y_mid=y_mid_center+int(np.sin(0.4*np.pi*i1+np.pi)*54*3)
    draw_circle(x_mid,y_mid,r,[0,255,255])

#连线
x_mid_center=270
y_mid_center=270
r=5
for i1 in range(5):
    x1=x_mid_center+int(np.cos(0.4*np.pi*i1+np.pi)*54*3)
    y1=y_mid_center+int(np.sin(0.4*np.pi*i1+np.pi)*54*3)
    x2=x_mid_center+int(np.cos(0.4*np.pi*i1+1.8*np.pi)*54*3)
    y2=y_mid_center+int(np.sin(0.4*np.pi*i1+1.8*np.pi)*54*3)
    draw_line(x1,x2,y1,y2)

#填色
x_mid=270
y_mid=270
r=54*3
fill_color_five_star(x_mid,y_mid,r,np.array([[1,0],[0,1]]))

# 画圆
x_mid=54*2
y_mid=54*10
r=5
draw_circle(x_mid,y_mid,r,[0,255,255])
x_mid=54*4
y_mid=54*12
draw_circle(x_mid,y_mid,r,[0,255,255])
x_mid=54*7
y_mid=54*12
draw_circle(x_mid,y_mid,r,[0,255,255])
x_mid=54*9
y_mid=54*10
draw_circle(x_mid,y_mid,r,[0,255,255])

# 画圆弧
x_mid=54*2
y_mid=54*10
r=54
draw_arc(x_mid,y_mid,r,[0,0,0])
x_mid=54*4
y_mid=54*12
draw_arc(x_mid,y_mid,r,[0,0,0])
x_mid=54*7
y_mid=54*12
draw_arc(x_mid,y_mid,r,[0,0,0])
x_mid=54*9
y_mid=54*10
draw_arc(x_mid,y_mid,r,[0,0,0])

#连线
x_mid_center=54*2
y_mid_center=54*10
r=54
theta=np.array([[-3,5],[-5,-3]]).T/np.sqrt(34)
print(theta)
for i1 in range(5):
    x1=np.cos(0.4*np.pi*i1+np.pi)*r
    y1=np.sin(0.4*np.pi*i1+np.pi)*r
    x2=np.cos(0.4*np.pi*i1+1.8*np.pi)*r
    y2=np.sin(0.4*np.pi*i1+1.8*np.pi)*r
    x1,y1=np.dot(theta,np.array([x1,y1]))
    x2,y2=np.dot(theta,np.array([x2,y2]))
    x1=int(x_mid_center+x1)
    x2=int(x_mid_center+x2)
    y1=int(y_mid_center+y1)
    y2=int(y_mid_center+y2)
    draw_line(x1,x2,y1,y2)
x_mid_center=54*4
y_mid_center=54*12
theta=np.array([[-1,7],[-7,-1]]).T/np.sqrt(50)
for i1 in range(5):
    x1=np.cos(0.4*np.pi*i1+np.pi)*r
    y1=np.sin(0.4*np.pi*i1+np.pi)*r
    x2=np.cos(0.4*np.pi*i1+1.8*np.pi)*r
    y2=np.sin(0.4*np.pi*i1+1.8*np.pi)*r
    print(x1,y1,x2,y2)
    x1,y1=np.dot(theta,np.array([x1,y1]))
    x2,y2=np.dot(theta,np.array([x2,y2]))
    print(x1,y1,x2,y2)
    x1=int(x_mid_center+x1)
    x2=int(x_mid_center+x2)
    y1=int(y_mid_center+y1)
    y2=int(y_mid_center+y2)
    draw_line(x1,x2,y1,y2)
x_mid_center=54*7
y_mid_center=54*12
theta=np.array([[2,7],[-7,2]]).T/np.sqrt(53)
for i1 in range(5):
    x1=np.cos(0.4*np.pi*i1+np.pi)*r
    y1=np.sin(0.4*np.pi*i1+np.pi)*r
    x2=np.cos(0.4*np.pi*i1+1.8*np.pi)*r
    y2=np.sin(0.4*np.pi*i1+1.8*np.pi)*r
    x1,y1=np.dot(theta,np.array([x1,y1]))
    x2,y2=np.dot(theta,np.array([x2,y2]))
    x1=int(x_mid_center+x1)
    x2=int(x_mid_center+x2)
    y1=int(y_mid_center+y1)
    y2=int(y_mid_center+y2)
    draw_line(x1,x2,y1,y2)
x_mid_center=54*9
y_mid_center=54*10
theta=np.array([[4,5],[-5,4]]).T/np.sqrt(41)
for i1 in range(5):
    x1=np.cos(0.4*np.pi*i1+np.pi)*r
    y1=np.sin(0.4*np.pi*i1+np.pi)*r
    x2=np.cos(0.4*np.pi*i1+1.8*np.pi)*r
    y2=np.sin(0.4*np.pi*i1+1.8*np.pi)*r
    x1,y1=np.dot(theta,np.array([x1,y1]))
    x2,y2=np.dot(theta,np.array([x2,y2]))
    x1=int(x_mid_center+x1)
    x2=int(x_mid_center+x2)
    y1=int(y_mid_center+y1)
    y2=int(y_mid_center+y2)
    draw_line(x1,x2,y1,y2)

#填色
x_mid=54*2
y_mid=54*10
r=54
fill_color_five_star(x_mid,y_mid,r,np.array([[-3,5],[-5,-3]]).T/np.sqrt(34))
x_mid=54*4
y_mid=54*12
fill_color_five_star(x_mid,y_mid,r,np.array([[-1,7],[-7,-1]]).T/np.sqrt(50))
x_mid=54*7
y_mid=54*12
fill_color_five_star(x_mid,y_mid,r,np.array([[2,7],[-7,2]]).T/np.sqrt(53))
x_mid=54*9
y_mid=54*10
fill_color_five_star(x_mid,y_mid,r,np.array([[4,5],[-5,4]]).T/np.sqrt(41))

# frame=(np.array([np.zeros(1749600),np.zeros(1749600),np.ones(1749600)*255],dtype='uint8').T).reshape((1080,1620,3))
# true_flag=(np.array([np.zeros(1749600),np.zeros(1749600),np.zeros(1749600)],dtype='uint8').T).reshape((1080,1620,3))
for y in range(162):
    # print(10*y,10*y-1080)
    frame=np.concatenate((true_flag[:,:10*y,:],frame[:,10*y:1620,:]),axis=1)
    inshow_and_write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
