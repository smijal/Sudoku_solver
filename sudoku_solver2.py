import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import glob


#Path
current_directory = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_directory , 'sudoku.png')
image = cv2.imread(image_path)
im_copy = image
print(image.shape)
image = cv2.resize(image,(1200,1200))
print(image.shape)
cell_size = image.shape[0]//9

#To create a new directory for saving the extracted cells from the sudoku table
current_directory = os.path.dirname(os.path.abspath(__file__))
final_directory = os.path.join(current_directory, r'extracted_cells')
if not os.path.exists(final_directory):
    os.makedirs(final_directory)

#Add neural network for digit recognition
#Open extracted_cells folder, read image one by one -> recognize -> write to the matrix
model_path = os.path.join(current_directory, 'saved_model/my_model12')
new_model = tf.keras.models.load_model(model_path)

#final matrix 
sudoku_matrix = np.zeros((9,9), dtype='int')
check_bool = np.zeros((9,9), dtype='int') #works like a boolean checksum -> when all cells are extracted becomes 9x9 matrix of Logic 1's
mask = np.ones((9,9), dtype='int')

#to make it grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

#to plot 2 images side by side
def plot_images(img1,img2, title1, title2):
    fig = plt.figure(figsize=[15,15])
    ax1 = fig.add_subplot(121)
    ax1.imshow(img1, cmap="gray")
    ax1.set(xticks=[], yticks=[], title=title1)

    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap="gray")
    ax2.set(xticks=[], yticks=[], title=title2)
    plt.show()

#for giving the correct name in the extracted cell files
def calculate_index(indexes):
    return str(indexes[1]) + str(indexes[0])

# to get rid of accidental frames in the extracted pictures and make it easier for recognition
def blendFrame(image):
    for i in range(3):
        image[i,:] = 0
        image[27-i,:] = 0
    for i in range(7):
        image[:,i] = 0
        image[:,27-i]=0
    return image

#resize to 28x28 to match the MNIST dataset pictures
def resize_tf(img):
    basewidth = 28
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    np_img=np.asarray(img)
    return np_img

#treshold some pixel for more clear view
def treshold(image):
    b = np.where(image < 140)
    c = np.where(image>=180)
    image[b] = 0
    image[c] = 255
    return image

#crop the center of the image to make the digit biger on the 28x28 frame
def cropND(image):
    image = image[:,13:126-13]
    image = image[13:126-13,:]
    return image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #make it gray
blur = cv2.GaussianBlur(gray, (3,3), 0) #blur to get rid of undesired pixels
inverted = np.invert(blur) #invert to black background -> white number
contours = cv2.Canny(blur, 20,200)  #contours
cnts, new = cv2.findContours(contours.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  #sort by intensity of contours
image_copy = image.copy() 
_ = cv2.drawContours(image_copy, cnts, -1, (255,0,255), 2)
#plot_images(blur,image_copy, 'thresh', 'contours')



cell = None
for c in cnts:
    perimeter = cv2.arcLength(c,True)
    edges_count = cv2.approxPolyDP(c, 0.02*perimeter, True)
    if(len(edges_count)==4):
        x,y,w,h = cv2.boundingRect(c)
        if(w>cell_size or h>cell_size or h<100 or w<100):
            print("Ignore")
        else:
            cell = image[y:y+h, x:x+w]
            cell=cropND(cell)
            cell = Image.fromarray(np.uint8(cell))
            resized = resize_tf(cell)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) #make it gray
            cell = np.invert(resized)
            cell = blendFrame(cell)
            cell = treshold(cell)
            indexes = [x//w, y//w]
            if(indexes[0]>8 or indexes[1]>8):
                # print("Index greater than 8")
                # print("Width= " + str(w))
                # print("Height = " + str(h))
                # print("Ceil x/w = "+ str(math.floor(x/w)))
                # print("Ceil y/h = "+ str(math.floor(y/h)))
                plot_images(cell,cell,str(x)+" x " + str(y), str(indexes))
            # elif(check_bool[indexes[0]][indexes[1]]==1):
            #     # print("Already scanned")
            else:
                check_bool[indexes[0]][indexes[1]] = 1
                # print("Index saved")
                cell_name = os.path.join(final_directory, calculate_index(indexes) + ".jpg")
                cv2.imwrite(cell_name, cell)
            if(np.array_equal(mask,check_bool)):
                print("All cells scanned")
                break
print("Checksum")
print(check_bool)

cells = []
files = glob.glob(os.path.join(final_directory,"*.jpg"))
files.sort()
for filename in files:
    img = cv2.imread(filename)
    img = rgb2gray(img)
    cells.append(img)
cells = np.array(cells)
cells.reshape(81,28,28)


#new_model.summary()
predictions = new_model.predict(cells)
i=0
for pred in predictions:
    #print("c=" + str(i))
    #print(pred)
    perc = max(pred)
    if(perc*100>95):
        value = np.where(pred==perc)
        sudoku_matrix[int(i/9),i%9]=value[0]
        #print(value)
        #print(perc*100)
        # plt.imshow(cells[i])
        # plt.show()
    #else:
        #print("Confidence under 90%")

    i+=1

print("Sudoku matrix: ")
print(sudoku_matrix)
to_fill = (sudoku_matrix==0)*1

empty_loc=[0,0]

def checkRow(sudoku_matrix, row_indx, num):
    row = sudoku_matrix[row_indx,:]
    if num in row:
        return True
    return False
def checkColumn(sudoku_matrix, col_indx, num):
    column = sudoku_matrix[:,col_indx]
    if num in column:
        return True
    return False

def checkBox(sudoku_matrix, row_indx, col_indx, num):
    sr = row_indx//3*3
    sc = col_indx//3*3
    box = sudoku_matrix[sr:sr+3,sc:sc+3]
    if num in box:
        return True
    return False

def isSafe(sudoku_matrix,ri,ci,num):
    return (not checkRow(sudoku_matrix,ri,num) and
           not checkColumn(sudoku_matrix,ci,num) and
           not checkBox(sudoku_matrix,ri,ci,num) )    

def findEmpty(sudoku_matrix,empty_loc):
    for i in range(9):
        for j in range(9):
            if(sudoku_matrix[i,j]==0):
                empty_loc[0]=i
                empty_loc[1]=j
                return True
    return False

#solution with backtracking
def solve(sudoku_matrix, empty_loc):
    if(not findEmpty(sudoku_matrix,empty_loc)):
        return True
    else:
        row=empty_loc[0]
        column = empty_loc[1]
    for num in range(1,10):
        if(isSafe(sudoku_matrix,row,column,num)):
            sudoku_matrix[row,column]=num
            if(solve(sudoku_matrix,empty_loc)):
                return True
            sudoku_matrix[row,column]=0
    return False

check_sol = solve(sudoku_matrix,empty_loc)
if(not check_sol):
    print("No solution found")
    exit()
print("Solved Matrix")
print(sudoku_matrix)

dig_directory = os.path.join(current_directory, r'digits')
files = glob.glob(os.path.join(dig_directory,"*.jpg"))
files.sort()
print(files)

def findDigit(files, digit):
    for filename in files:
        l = len(filename)
        name=filename[l-6:]
        if(name == "d"+str(digit) + ".JPG"):
            d=cv2.imread(filename)
            d=cv2.resize(d,(100,100))
            # cv2.imshow("d",d)
            return d

def writeSolution(sudoku_matrix, to_fill, image_copy):
    for i in range(9):
        for j in range(9):
            if(to_fill[i,j]==1):
                d=findDigit(files,sudoku_matrix[i,j])
                image_copy[i*130+30:i*130+30+100, j*130+30:j*130+100+30,:]  = d

writeSolution(sudoku_matrix,to_fill,im_copy)
plot_images(image,im_copy,'original','solution')

cv2.waitKey(0)
cv2.destroyAllWindows()

