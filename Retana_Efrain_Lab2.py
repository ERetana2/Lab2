# -*- coding: utf-8 -*-
"""
@Assignment: Lab 2
@Professor: Olac Fuentes
@TA: Oscar Galindo
@Author: Efrain Retana
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def read_image(imagefile):
    # Reads image in imagefile and returns color and gray-level images
    #
    img = (plt.imread(imagefile)*255).astype(int)
    img = img[:,:,:3]  # Remove transparency channel
    img_gl = np.mean(img,axis=2).astype(int)
    return img, img_gl

def draw_rectangle(r,c,h,w,color): # (r,c) coordinates for rectangle, h,w for height 
    rect_points = np.array([ [r,c],[r+w,c],[r+h,c+w],[r,c+w],[r,c] ])
    ax1.plot(rect_points[:,0],rect_points[:,1],linewidth = 2.0,color = color)
    ax2.plot(rect_points[:,0],rect_points[:,1],linewidth = 2.0,color = color)
    
def brightest_pixel(I,h,w):
    imgMax = 0.0
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            currPixel = 0
            for x in range(3):
                currPixel += I[i,j,x]
            if currPixel > imgMax:
                imgMax = currPixel
                max_x_val = j
                max_y_val = i
    draw_rectangle(max_x_val - h/2,max_y_val-w/2,h,w,'red')
    
def brightest_region(img_gl,h,w):
    max_region = 0
    for i in range(img_gl.shape[0] - h):
        for j in range(img_gl.shape[1] - w):
            curr_region = 0
            for height in range(i,h + i ):
                for width in range(j,w + j):
                   curr_region += img_gl[height,width]
                if curr_region > max_region:
                    max_region = curr_region
                    max_x_val = j
                    max_y_val = i
    draw_rectangle(max_x_val,max_y_val,h,w,'green')
    
def brightest_region1_2(img_gl,h,w):
    max_region = 0
    for i in range(len(img_gl)-h):
        for j in range(len(img_gl[i])-w):
            curr_region = np.array(img_gl[i:h+i,j:w+j])
            sum_region = np.sum(curr_region)
            if sum_region > max_region:
                max_region = sum_region
                max_x_val = j
                max_y_val = i
    draw_rectangle(max_x_val,max_y_val,h,w,'red')
def integral(img_gl,h,w):
    # create a temporary variable to add a row and column of zeros to the current
    # image matrix
    integral_img =  np.cumsum(img_gl,1)
    integral_img = np.cumsum(integral_img,0)
    integral_img = np.insert(integral_img,0,0,0)
    integral_img = np.insert(integral_img,0,0,1)
    return integral_img
    
#    for i in range(len(integral_img)):
#        for j in range(len(integral_img[0])):
#            curr_row = integral_img[i,:]
#            curr_row = curr_row[:j+1]
#            integral_img[i][j] = integral_img[i-1][j] + np.sum(curr_row)
        
def brightest_region2_1(img_gl,h,w):
    
    integral_img = integral(img_gl,h,w)
    region_max = 0
    for r in range(len(integral_img) - h):
        curr_region = 0
        for c in range(len(integral_img[0]) -w):
            topLeft = integral_img[r][c]
            botLeft = integral_img[r+h][c]
            botRight = integral_img[r+h][c+w]
            topRight = integral_img[r][c+w]
            curr_region = botRight - topRight - botLeft + topLeft
            if curr_region > region_max:
                region_max = curr_region
                max_x_val = c
                max_y_val = r
    draw_rectangle(max_x_val + 1,max_y_val + 1,h,w,'yellow')
      
def brightest_region2_2(img_gl,h,w):
    integral_img = integral(img_gl,h,w)
    
    botRight = integral_img[h:,w:]
    topRight = integral_img[h:,:len(integral_img) - w]
    botLeft = integral_img[:len(integral_img) - h,w:]
    topLeft = integral_img[:len(integral_img) - h,:len(integral_img) - w]
    sum = botRight - topRight - botLeft + topLeft
    max = np.max(sum)
    
    sum_index = np.argwhere(sum[:]==max)
    topleft_val = topLeft[sum_index[0][0]][sum_index[0][1]]
    integral_index = np.argwhere(integral_img[:]==topleft_val)
    
    x,y = integral_index[0][1],integral_index[0][0]
    draw_rectangle(x,y,h,w,'yellow')
            


if __name__ == "__main__":

    img_dir = '.\\solar images\\' # Directory where imagea are stored
    
    img_files = os.listdir(img_dir)  # List of files in directory
    
    plt.close('all')
    
    for file in img_files:
        print(file)
        if file[-4:] == '.png': # File contains an image
            img, img_gl = read_image(img_dir+file)
    
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            ax1.imshow(img)                  #Display color image
            ax2.imshow(img_gl,cmap='gray')   #Display gray-leval image
            #brightest_pixel(img)
            #brightest_region(img_gl,50,50)]
            brightest_region2_2(img_gl,50,50)
            #brightest_region2_1(img_gl,50,50)
            plt.show()
            


