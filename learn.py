import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import preprocessing

IMAGE_PATH = "images"

#A function that outputs the first 25 faces from the given face matrix
def faces_plot(matrix, sz, name):
    fig, axes = plt.subplots(5,5,sharex=True,sharey=True,figsize=(10,10))
    for i in range(25):
        axes[i%5][i//5].imshow(np.resize(matrix[i], (sz[0], sz[1])), cmap="gray")
    plt.savefig(name)

print("Starting image processing and learning...")

#Project directory
project_dir = os.path.dirname(os.path.abspath(__file__))

#The name of the photo folder is appended
images_dir = os.path.join(project_dir, IMAGE_PATH)

#The dictionary is designed to store photos by path
faces = {}
#Uniform size of all photos is selected
size = (250,250)

for dirs, paths, images in os.walk(images_dir):
	for image in images:
		if image.endswith("jpg") or image.endswith("png"):
            #Selects photo file names as labels
			image_label = os.path.basename(dirs).replace(" ", "-")
            
			#Photos are converted to grayscale and accessible size
			grey_image = Image.open(os.path.join(dirs, image)).convert("L").resize(size)

            #Faces are stored in a dictionary along with a folder and file name
			faces[image_label+"\\"+image] = grey_image

#We count the number of individual individuals and the number of photos available
people = set(paths.split("\\")[0] for paths in faces.keys())

#Face matrix
faces_matrix = []

for key,val in faces.items():
    #The training reduced dimension values are added separately to another list
	faces_matrix.append(np.array(val, "float64").flatten())

#List objects are cast to an array 
faces_matrix = np.array(faces_matrix)

#Table of primary Faces
faces_plot(faces_matrix, size, 'beginning_data.png')

#Calculating face average
mean_face = np.sum(faces_matrix, axis=0, dtype='float64')/len(faces)

#The derived average is saved
fig, axes = plt.subplots(1,1,figsize=(7,7))
plt.imshow(np.resize(mean_face, (size[0],size[1])),cmap='gray')
plt.title('Visų veidų vidurkis')
plt.savefig('mean.png')

#Data centered (faces mean subtracted)
for i in range(len(faces_matrix)):
    faces_matrix[i] = faces_matrix[i] - mean_face

#Table of centered faces
faces_plot(faces_matrix, size, 'centralized_data.png')

#The covariance matrix is found
C = faces_matrix.dot(faces_matrix.T)/len(faces)

#A function that finds eigen values and eigen vectors
eigenvalues, eigenvectors = np.linalg.eig(C)

#We order eigen vectors by eigen values
list1, list2 = zip(*sorted(zip(eigenvalues, eigenvectors)))

#We search for eigen faces by multiplying eigen vectors with centered data
eigenfaces = (faces_matrix.T @ eigenvectors).T

eigenfaces = preprocessing.normalize(eigenfaces)

#Eigen face plot
faces_plot(eigenfaces, size, 'eigenfaces.png')

#The stored data is necessary for identification
np.save('faces_matrix', np.array(faces_matrix))
np.save('mean_face', np.array(mean_face))
np.save('eigenfaces', np.array(eigenfaces))

with open('about_learn.txt', 'w',encoding="utf-8") as f:
    f.write("Number of people in learning data: " + str(len(people)) + "\n")
    f.write("Number of photos in training data: " + str(len(faces)) + "\n")
    f.write("mean_face.npy -> an array of the averaged face is saved"  + "\n")
    f.write("faces.npy -> an array of faces saved by path"  + "\n")
    f.write("eigenfaces.npy -> arrays of real faces in the training data" + "\n")
    f.write("beginning_data.png -> 25 examples of initial training data"  + "\n")
    f.write("mean.png -> the averaged face of the training data"  + "\n")
    f.write("centralized_data.png -> mokymosi duomenų centruotų veidų lentelė"  + "\n")
    f.write("eigenfaces.png -> learning data centered face table" + "\n")
    f.write("{{skaičius}}.png -> test photos with the most matching photo")

print("Learning finished. Check readme_learn.txt for more information.")
