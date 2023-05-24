import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

#Function performing recognition and calculation of weights
def recognising(image, sz, mean_face, eigenfaces_num, eigenfaces, faces_matrix, name):
    #The average of all faces is subtracted from the photo
    image_mean = np.reshape(image, (sz[0]*sz[1])) - mean_face
    #The image is projected into the face space and the 
    # calculated weights are collected into an array
    omega = eigenfaces[:eigenfaces_num].dot(image_mean)

    #Defines the variable of the smallest Euclidean distance found and its index
    smallest_dist = None 
    sd_index = None

    for k in range(len(faces_matrix)):
        #The weight of the compared photo is calculated
        omega_k = eigenfaces[:eigenfaces_num].dot(faces_matrix[k])
        #The weights of the comparison photo are subtracted from
        # the weights of the test photo
        difference = omega - omega_k
        #Completing the Euclidean distance calculation
        epsilon_k = math.sqrt(difference.dot(difference))
        #Searching for the shortest distance
        if smallest_dist == None:
            smallest_dist = epsilon_k
            sd_index = k
        if smallest_dist > epsilon_k:
            smallest_dist = epsilon_k
            sd_index = k
     
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    axes[0].imshow(np.reshape(image_mean + mean_face, (sz[0],sz[1])), cmap='gray')
    axes[0].set_title("Centruota testinė nuotrauka")
    axes[1].imshow(np.reshape(faces_matrix[sd_index] + mean_face, (sz[0],sz[1])), cmap='gray')
    axes[1].set_title("Panašiausias atitikmuo su svoriu: " + str(round(smallest_dist,2)))
    plt.savefig(name)

    return round(smallest_dist,2) <= 2424.0

#A function that performs facial reconstruction
def reconstruct(image, sz, mean_face, eigenfaces):
    fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(15,15))
    eigenface_num = [1,3, 5, 10, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1100]
    for i, num in enumerate(eigenface_num):
        #The average of all faces is subtracted from the photo
        image_mean = np.reshape(image, (sz[0]*sz[1])) - mean_face
        #The image is projected into the face space and the calculated
        # weights are collected into an array
        omega = eigenfaces[:num].dot(image_mean)
        #A weighted sum of eigen faces is calculated, where the weights are given by omega
        F = mean_face + omega.dot(eigenfaces[:num])
        #Reconstructed by adding F to the middle face
        reconstruct = mean_face + F
        axes[i//4][i%4].imshow(np.resize(reconstruct, (sz[0], sz[1])), cmap="gray")
        axes[i//4][i%4].set_title("Tikrinių veidų skaičius: " + str(num))    
    plt.savefig("reconstructed.png")

size = (250,250)
IMAGE_DIR = "asmenines"
#Project folder
project_dir = os.path.dirname(os.path.abspath(__file__))

#The name of the photo folder is appended
images_dir = os.path.join(project_dir, IMAGE_DIR)
#The dictionary is designed to store photos by path
faces = {}
#Uniform size of all photos is selected
size = (250,250)

for image in os.listdir(images_dir):
    if image.endswith("jpg") or image.endswith("png"):
        #Selects photo file names as labels
        image_label = os.path.basename(images_dir).replace(" ", "-")
                    
        #Photos are grayed out
        grey_image = Image.open(os.path.join(images_dir, image)).convert("L").resize(size)

        #Faces are stored in a dictionary along with a folder and file name
        faces[image_label+"\\"+image] = grey_image


#List of test photos
test_images = []

for key,val in faces.items():
    #The training reduced dimension values are added separately to another list
	test_images.append(np.array(val, "float64").flatten())
        
#List objects are cast to an array
test = np.array(test_images)

#The data being loaded is from the training program
faces_matrix = np.load('faces_matrix.npy')
mean_face = np.load('mean_face.npy')
eigenfaces = np.load('eigenfaces.npy')
#Array to store all face weights
weight = []

for count, image in enumerate(test):
    rec = recognising(image, size, mean_face, 50, eigenfaces, faces_matrix, (str(count)+".png"))
    weight.append(rec)
    
with open('non-existing_recognised.txt', 'w',encoding="utf-8") as f:
    f.write("Atpažinti veidai:" + str(sum(weight)) + "\n")
       
for count, image in enumerate(test):
    rec = reconstruct(image, size, mean_face, eigenfaces)
    