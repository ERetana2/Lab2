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
import time

def read_image(imagefile):
    # Reads image in imagefile and returns color and gray-level images
    #
    img = (plt.imread(imagefile)*255).astype(int)
    img = img[:,:,:3]  # Remove transparency channel
    img_gl = np.mean(img,axis=2).astype(int)
    return img, img_gl
#-------------------------------------------
def draw_rectangle(ax,r,c,h,w,color): # (r,c) coordinates for rectangle, h,w for height 
    rect_points = np.array([ [r,c],[r+w,c],[r+h,c+w],[r,c+w],[r,c] ])
    ax.plot(rect_points[:,0],rect_points[:,1],linewidth = 2.0,color = color)
#-----------------------------------------------------------------------------   
# PROBLEM 1
def brightest_pixel(ax,I,h,w):
    imgMax = 0.0
    #Iterate through all the pixels in the img to find brightest pixe;
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            currPixel = 0
            for x in range(3): # iterates through the 3 channels in the matrix, (r,g,b)
                currPixel += I[i,j,x] # add all channels for current pixel
            # if max is found, save x and y values and set max to current pixel
            if currPixel > imgMax:
                imgMax = currPixel
                max_x_val = j
                max_y_val = i
    draw_rectangle(ax,max_x_val - h/2,max_y_val-w/2,h,w,'red')
#-------------------------------------------------------------------
# PROBLEM 2
def brightest_region(ax,img_gl,h,w):
    # compute the brightest region inside of the matrix given h x w region using 
    # 4 for loops
    max_region = 0
    # first two for loops iterating through the rows and columns of matrix
    for i in range(img_gl.shape[0] - h):
        for j in range(img_gl.shape[1] - w):
            curr_region = 0
            # second two for loops calculating the sum of current region as in problem 1
            # then comparing to current max
            for height in range(i,h + i ):
                for width in range(j,w + j):
                   curr_region += img_gl[height,width]
                  # save max, and save x,y values for max region
                if curr_region > max_region:
                    max_region = curr_region
                    max_x_val = j
                    max_y_val = i
    draw_rectangle(ax,max_x_val,max_y_val,h,w,'green')
#--------------------------------------------------------
# PROBLEM 1.2
def brightest_region1_2(img_gl,h,w):
    # Compute brightest region by cutting 2 for loops and utilizing slicing
    max_region = 0
    for i in range(len(img_gl)-h): # iterate through rows 
        for j in range(len(img_gl[i])-w): # then columns
            #create an array containing a matrix of the current region
            curr_region = np.array(img_gl[i:h+i,j:w+j])
            sum_region = np.sum(curr_region) # sum the whole region
            # compare current sum to max region, then save max and x,y values
            if sum_region > max_region:
                max_region = sum_region
                max_x_val = j
                max_y_val = i
    draw_rectangle(max_x_val,max_y_val,h,w,'red')
#-------------------------------------------------------
def integral(ax,img_gl,h,w):
    # compute integral image of matrix by doing cumulative sum function
    # and adding a row and column of 0s
    integral_img =  np.cumsum(img_gl,1)
    integral_img = np.cumsum(integral_img,0)
    integral_img = np.insert(integral_img,0,0,0)
    integral_img = np.insert(integral_img,0,0,1)
    return integral_img
#--------------------------------------------
# PROBLEM 2.1     
def brightest_region2_1(ax,img_gl,h,w):
    integral_img = integral(img_gl,h,w)
    region_max = 0
    # iterate through rows and columns of each h x w square then obtain the top left
    #,bottom left, top right, and bottom left elements to compute the current region using
    # A - B - C + D
    for r in range(len(integral_img) - h):
        curr_region = 0
        for c in range(len(integral_img[0]) -w):
            topLeft = integral_img[r][c]
            botLeft = integral_img[r+h][c]
            botRight = integral_img[r+h][c+w]
            topRight = integral_img[r][c+w]
            curr_region = botRight - topRight - botLeft + topLeft
            if curr_region > region_max: #once max is obtained, save max and x,y values
                region_max = curr_region
                max_x_val = c
                max_y_val = r
    draw_rectangle(ax,max_x_val + 1,max_y_val + 1,h,w,'yellow')
#--------------------------------------------------------------
# PROBLEM 2.2
def brightest_region2_2(ax,img_gl,h,w):
    integral_img = integral(ax,img_gl,h,w)
    #obtain 4 arrats containing each every bottom right, left, top right and left point for every
    # region inside of the integral img using slicing
    botRight = integral_img[h:,w:]
    topRight = integral_img[h:,:len(integral_img) - w]
    botLeft = integral_img[:len(integral_img) - h,w:]
    topLeft = integral_img[:len(integral_img) - h,:len(integral_img) - w]
    # compute sum A-B-C+D then find the max number in all the set of arrays
    sum = topLeft - topRight - botLeft + botRight
    max = np.max(sum)
    
    sum_index = np.argwhere(sum[:]==max) # return the index of the sum that equals max
    topleft_val = topLeft[sum_index[0][0]][sum_index[0][1]] # obtain top left value at the indices 
                                                            # of the max sum
    integral_index = np.argwhere(integral_img[:]==topleft_val) # search whole integral
    # for value that equals top left value, plot a rectangle, which signals max/
    # brightest region
    draw_rectangle(ax,integral_index[0][1],integral_index[0][0],h,w,'yellow')
#-------------------------------------------------------------------------------------       


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
            plt.show()
    #brightest_pixel(ax,img)
    #brightest_region(aximg_gl,50,50)]
    #brightest_region2_2(ax,img_gl,50,50)
    #brightest_region2_1(ax,img_gl,50,50)


