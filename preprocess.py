import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def loadImages(path, category):
    '''
    function to load folder into arrays
    path: string, image files path
    category: string, category of the Mexican pottery
    return: array, list of image file paths
    '''
    # Put files into lists and return them as one list of size 4
    image_files = sorted([os.path.join(path, category, file)
                          for file in os.listdir(path + category) 
                          if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')])
 
    return image_files

def isSquare(image):
    '''
    function to test wether the image is square shaped
    image: numpy array represents images
    return: True, if the image shape is square; else, False
    '''
    return image.shape[0] == image.shape[1]

def crop(image):
    '''
    function to crop the image into square
    image: numpy array represents images
    return: cropped image
    '''
    height = image.shape[0]
    width = image.shape[1]
    if height < width:
        a = height//2
        cropped_image = image[:, width//2-a: width//2+a]
    else:
        a = width//2
        cropped_image = image[height//2-a: height//2+a, :]
    
    if isSquare(cropped_image):
        return cropped_image
    else:
        if cropped_image.shape[0] > cropped_image.shape[1]:
            cropped_image = cropped_image[:-1, :, :]
        return cropped_image

def resize(image, dsize):
    '''
    function to resize the image to the same, required size
    image: numpy array represents images
    dsize: tuple, desired size
    return: scaled image
    '''
    return cv2.resize(image, dsize)
    
def get_data(image_path, category, dsize=(256, 256)):
    '''
    function to get and preprocess data into the same shape required for style GAN
    image_path: string, image path
    category: category of Mexican pottery
    return list of numpy array represents each image
    '''
    image_files = loadImages(image_path, category)
    images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in image_files]
    preprocessed_images = []
    for image in images:
        if not isSquare(image):
            image = crop(image)
        image = resize(image, dsize)
        if len(image.shape) == 3 and image.shape[2] == 3:
            preprocessed_images.append(image)
    return np.array(preprocessed_images)

def create_image(image_path, category):
    '''
    function to save the preprocessed data
    image_path: string, image path
    category: category of Mexican pottery
    '''
    images = get_data(image_path, category)
    for i in range(len(images)):
        cv2.imwrite(f'preprocessed_images/{category}/{category}_{i}.png', images[i])
        if i % 30 == 0:
            print(f'creating {category} images')

if __name__=="__main__":
    image_path = 'raw_images/'
    categories = ['Cazuela', 'Copalero', 'Tinaja']
    for category in categories:
        create_image(image_path, category)
