import cv2
feather=cv2.imread("./images/img1.jpeg")
rgb=cv2.cvtColor(feather,cv2.COLOR_BGR2RGB)

x=int(input("enter the x value:"))
y=int(input("enter the y value:"))
print("RGB values are:")
print(rgb[x,y,0],rgb[x,y,1],rgb[x,y,2])
#if we dont use rgb then then we print the last line using feather variable itself
#as it reads in brg so we are converting it to rgb using rgb variable