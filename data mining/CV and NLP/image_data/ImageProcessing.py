#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 17:52:40 2021

@author: sai
"""


from matplotlib import pyplot as plt
from skimage import io
import numpy as np
from skimage import color, img_as_float, feature
from skimage.filters import threshold_otsu 
from skimage.util import random_noise
from scipy import ndimage
from skimage.segmentation import slic
from skimage.transform import hough_line, hough_line_peaks
import os


class ImageProcessing(object):
    
    def __init__(self):
        
        self.Avengers = io.imread('avengers_imdb.jpg')
        self.bush_house = io.imread("bush_house_wikipedia.jpg")
        self.forestry = io.imread("forestry_commission_gov_uk.jpg")
        self.rolland = io.imread("rolland_garros_tv5monde.jpg")
        if not os.path.exists('outputs'):
            os.makedirs('outputs')


    def Task2Question1(self):
        
        print(f" Type of array : {type(self.Avengers)}")
        print(f" dtype : {self.Avengers.dtype}")
        print(f"Size of the Image : {self.Avengers.shape}")
        print(f" Min pixel value : {self.Avengers.min()}  Max pixel value : {self.Avengers.max()}")
        
        img = self.Gray_scale(self.Avengers)
        binary = self.Binary(img)
        
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True , figsize=(30,10), num = "Avengers")

        ax0.imshow(self.Avengers)
        ax0.set_title('Original Image')
        ax0.axis('off')

        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Gray Scale')
        ax1.axis('off')
        
        ax2.imshow(binary, cmap=plt.cm.gray)
        ax2.set_title('Binary')
        ax2.axis('off')
        
        
        plt.savefig('outputs/Avengers.png')
        
        
    def Gray_scale(self,image):
        
        img = image.copy()
        img = img_as_float(img)
        gray = img @ [0.2126,0.7152,0.0722]
        return gray

    def Binary(self,gray):
        Gray = gray.copy()
        thresh = threshold_otsu(Gray)
        binary = Gray > thresh
        return binary
    
    # Gaussian Random Noise
    def Gaussian_Random_Noise(self,image):
        img = image.copy()
        img = img_as_float(img)
        img = random_noise(img , mode="gaussian" ,var=0.1)
        return img
    
    # Gaussian Mask to Filter , sigma = 1 
    def Gaussian_Filter(self,NoiseImage, sigma):
        Noise = NoiseImage.copy()
        filtered_img = ndimage.gaussian_filter(Noise, sigma=sigma, mode='reflect')
        return filtered_img

    # Smoothing Mask 


    def SmoothingMask(self, NoiseImage):
        mean_kernel_9 = np.full(shape=(9,9) ,fill_value=1/81, dtype=np.float32)
        Noise = NoiseImage.copy()
        channels = {}
        for i in range(3):
            #filteredImage = ndimage.correlate(Noise[:,:,i], mean_kernel_9, mode='reflect')
            filteredImage = ndimage.convolve(Noise[:,:,i], mean_kernel_9, mode='reflect')
            channels[i] = filteredImage
        filtered_image = np.dstack((channels[0],channels[1],channels[2]))
        return filtered_image
    
    def Task2Question2(self):
        
        Noise_image = self.Gaussian_Random_Noise(self.bush_house)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True , figsize=(20,10 ))
        
        ax1.imshow(self.bush_house)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(Noise_image)
        ax2.set_title('Noisey Image')
        ax2.axis('off')
        
        plt.savefig('outputs/NoiseImage.png')
        
        
        filtered_img_1 = self.Gaussian_Filter(Noise_image,1)
        filtered_img_3 = self.Gaussian_Filter(Noise_image,3)
        filtered_image_smooth = self.SmoothingMask(Noise_image)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True , figsize=(30,10 ))

        ax1.imshow(filtered_img_1)
        ax1.set_title('Gaussian Filter sigma = 1')
        ax1.axis('off')
        
        ax2.imshow(filtered_img_3)
        ax2.set_title('Gaussian Filter sigma = 3')
        ax2.axis('off')
        
        ax3.imshow(filtered_image_smooth)
        ax3.set_title('Average Smoothing Filter')
        ax3.axis('off')
        
        plt.savefig('outputs/BlurComparison.png')
        
        
    def K_Means(self,image,numOfSegments):
        segments = slic(image, n_segments=numOfSegments, compactness=10, start_label=1)
        return segments
    
    def Task2Question3(self):
        segments = self.K_Means(self.forestry,5)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True , figsize=(20,10 ))
        
        ax1.imshow(self.forestry)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(segments)
        ax2.set_title('Segments')
        ax2.axis('off')
        
        plt.savefig('outputs/KMeans.png')
        
    def Canny(self,image,sigma):
        img = image.copy()
        img = img_as_float(img)
        gray = color.rgb2gray(img)
        edges = feature.canny(gray, sigma=sigma)
        return edges
    
    def HoughLine(self,edge):
        # Set a precision of 0.5 degree.
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        h, theta, d = hough_line(edge, theta=tested_angles)
        return h , theta , d
    
    def HoughLinePeaks(self,h, theta , d):
        h, q, d = hough_line_peaks(h, theta, d)
        return h, q, d
    
    def Task2Question4(self):
        
        edges1 = self.Canny(self.rolland,1)
        edges2 = self.Canny(self.rolland,2)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True , figsize=(30,10 ))
        
        ax1.imshow(edges1)
        ax1.set_title('Canny Edge Detect sigma = 1')
        ax1.axis('off')
        
        ax2.imshow(edges2)
        ax2.set_title('Canny Edge Detect sigma = 2')
        ax2.axis('off')
        
        ax3.imshow(self.rolland)
        ax3.set_title('Original Image')
        ax3.axis('off')
        
        plt.savefig('outputs/Canny.png')
        
        
        HL_HoughSpace , Theta , Distances = self.HoughLine(edges1)
        h, q, d = self.HoughLinePeaks(HL_HoughSpace , Theta , Distances)
        angle_list=[]  #Create an empty list to capture all angles
        
        
        fig, ax = plt.subplots(1, 1, sharey=True , figsize=(15,15 ))
        
        ax.imshow(HL_HoughSpace)
        ax.set_title('Hough Space Colour')
        ax.axis('off')
        
        plt.savefig('outputs/HoughSpace.png')

        print(f"Number of angles between lines : {len(q)}")
        print(f"Number of lines : {len(d)}")
        
        
        # Generating figure 1
        fig, axes = plt.subplots(1, 2, figsize=(20, 10),  squeeze=False)
        ax = axes.ravel()
        
        ax[0].imshow(edges1, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()
        
        
        
        ax[1].imshow(edges1, cmap='gray')
        
        origin = np.array((0, edges1.shape[1]))
        
        for _ , angle, dist in zip(*self.HoughLinePeaks(HL_HoughSpace , Theta , Distances)):
            angle_list.append(angle) #Not for plotting but later calculation of angles
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            ax[1].plot(origin, (y0, y1), '-r')
            
        ax[1].set_xlim(origin)
        ax[1].set_ylim((edges1.shape[0], 0))
        ax[1].set_axis_off()
        ax[1].set_title('Detected lines')
        
        plt.tight_layout()
        plt.savefig('outputs/ComparisonHoughTransform.png')
        
        
        fig, ax = plt.subplots(1, 1, sharey=True , figsize=(15,15 ))


        ax.imshow(np.log(1 + HL_HoughSpace),
                     extent=[np.rad2deg(Theta[-1]), np.rad2deg(Theta[0]), Distances[-1], Distances[0]],
                      aspect=1/1.5)
        ax.set_title('Hough transform')
        ax.set_xlabel('Angles (degrees)')
        ax.set_ylabel('Distance (pixels)')
        
        plt.savefig('outputs/DetailedHoughSpace.png')
        
        
    def main(self):
        print()
        print("------------------- Question 2.1 -------------------------------")
        print()
        
        self.Task2Question1()
        
        print()
        print("------------------- Question 2.2 -------------------------------")
        print()
        
        self.Task2Question2()
        
        print()
        print("------------------- Question 2.3 -------------------------------")
        print()
        
        self.Task2Question3()
        
        print()
        print("------------------- Question 2.4 -------------------------------")
        print()
        
        self.Task2Question4()
        
if __name__ == "__main__":

    ImageProcessing = ImageProcessing()
    ImageProcessing.main()
        






















