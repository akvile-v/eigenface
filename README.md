
# Eigenface algoritm 

## Goal

The purpose of the program is to calculate a threshold value from photos of a selected size, indicating
  whether the face seen in the photo is recognizable and to check the accuracy of the selected value
   and face recognition performance for different face rotation and lighting conditions implementing [Eigenface algorithm](https://doi.org/10.1162/jocn.1991.3.1.71).

## Programs and their usage

- ### learn.py:
    - the program calculates the eigen faces, face average and face matrix of the given training data. 
    - The saved data is used in the recognition program.
    - Depending on the training images, the IMAGE_PATH variable must be changed
    - The names of the stored files can also be changed, but it depends on the user himself.

- ### recognise.py:
    -  the program performs face recognition, weight calculation and face reconstruction.

## Data bases used for learning and testing

The input of the program is the training and testing images. The program was used for training and testing
part of the on-demand [Chicago Face Database](https://www.chicagofaces.org/) photo database dedicated to
 for non-commercial research purposes. The database features high resolutions
  standardized photographs of men and women of various ethnic backgrounds, aged 17 to 65 years. For training
   select 589 individuals, some of whom have more than one photo (1103 photos in total).
   For testing, 100 photos were selected for two cases: the face exists in the training data,
    but the photo was not used in the algorithm calculations, and the face does not exist in the training data.
 The size of the used photos in the program code has been reduced and standardized to 250px x 250px.

From earlier works, there was used another databases. A portion of the freely available [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) photo database, which is dedicated to face recognition research in natural environments, was used for training and testing the program. 110 persons with 5 different photos each were selected for training. For the test data, 10 photos were selected for three separate cases: the face does not exist in the photo, the face is unknown to the algorithm, and the face exists in the training data. Size of used photos - 250px x 250px.

## Usage

Programs should be used depending on scope and goal of research.

Recognition can be tested:
```
python recognise.py
```
after changing needed data inside the program.

## Scope of usage

Program was used in bachelor thesis practical part.