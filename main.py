import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

#parameters may change from one image to another, depends on the desired results
SIGMA = 0.8
RESOLUTION = 1
VOTES = 150


def EdgeDetection(im,sigma):
    #following the Canny algoritm for Edge Detection:
    blurred = cv2.GaussianBlur(grayIm, [3, 3], sigma)  # smoothing the image - removing noise
    m = np.median(blurred)
    lower = int(max(0, (1.0-SIGMA)*m))      #automaticly finding lower bound for Canny algorithm
    upper = int(min(255, (1.0-SIGMA)*m))    #automaticly finding upper bound for Canny algorithm
    edges = cv2.Canny(blurred, lower, upper)
    return edges


def FindLines(edges, ogIm):
    #using Hough Transform:
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edges = cv2.erode(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    lines = cv2.HoughLinesP(edges, RESOLUTION, np.pi/180, VOTES,None,100,10) #find the extremes of the lines in the picture
    if lines is None:  #checking if no lines were detected
        print('ERROR! did not detect any lines!')
        return 0, 0
    allLines=np.zeros([lines.shape[0], 6])
    for i in range(0, lines.shape[0]): #drawing the lines on the picture
        line = lines[i][0]
        cv2.line(ogIm, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
        allLines[i, 0] = line[0] #Xstart
        allLines[i, 1] = line[1] #Ystart
        allLines[i, 2] = line[2] #Xend
        allLines[i, 3] = line[3] #Yend
        allLines[i, 4] = round(np.rad2deg(math.atan2((line[3]-line[1]), (line[2]-line[0])))) #the angle the line creates with the x asix
        allLines[i, 5] = np.linalg.norm([line[2]-line[0], line[3]-line[1]]) #the length of the line
    return ogIm, allLines

def FindLongestPair(all, originalIm):
    current =np.array([])
    all[all[:, 4].argsort()]
    i = 0
    longest = np.zeros(6)
    secondLongest = np.zeros(6)
    found = False
    for theta in range(-90, 90):
        while i < all.shape[0] and all[i, 4] == theta:
            current = np.append(current, all[i])
            i += 1
        if current.any():
            current = current.reshape((-1, 6))
            if current.shape[0] >= 2:
                current[current[:, 5].argsort()]
                if found and secondLongest[5] > current[1, 5]:
                    secondLongest = current[1]
                    longest = current[0]
                else:
                    secondLongest = current[1]
                    longest = current[0]
                    found = True
                current = np.array([])
    if found: #if found, draw the lines in the image
        cv2.line(originalIm, (int(longest[0]), int(longest[1])), (int(longest[2]), int(longest[3])), (0, 0, 255), 2)
        cv2.line(originalIm, (int(secondLongest[0]), int(secondLongest[1])), (int(secondLongest[2]), int(secondLongest[3])), (0, 0, 255), 2)
        return originalIm
    else:
        print('did not find any pairs')
        return 0



if __name__ == '__main__':
    originalIm = cv2.imread('images/image2.jpg')
    copy2 = cv2.imread('images/image2.jpg')
    copy3 = cv2.imread('images/image2.jpg')
    grayIm = cv2.cvtColor(originalIm, cv2.COLOR_BGR2GRAY) #converting image to grayscale
    edges = EdgeDetection(grayIm, SIGMA)
    linesImage, all = FindLines(edges, originalIm)
    two = FindLongestPair(all, copy2)
    cv2.imshow("The_Lines: ", two)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(copy3)
    axs[0, 0].set_title('Original Image: ')
    axs[0, 1].imshow(edges, cmap='gray')
    axs[0, 1].set_title('Edge Image: ')
    axs[0, 1].sharex(axs[0, 0])
    axs[1, 0].imshow(linesImage,cmap='Accent')
    axs[1, 0].set_title('All lines: ')
    axs[1, 1].imshow(two)
    axs[1, 1].set_title('Two longest lines in the image: ')
    fig.tight_layout()
    plt.show()
    cv2.waitKey(0)






