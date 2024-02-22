import glob
import config
import cv2
import numpy as np
import random

seed = np.random.randint(1)
random.seed(seed)
np.random.seed(seed)

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img




def clahe(img, cl=2.0 ):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = crop_image_from_gray(img)
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(50, 50))
    img = clahe.apply(img)
    return img

def clahe_green(img, cl=10.0):
    img = crop_image_from_gray(img)
    green = img[:, :, 1]
    clipLimit = config.CL
    tileGridSize = (int(config.GRID), int(config.GRID))
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cla = clahe.apply(green)
    img = cv2.merge((cla, cla, cla))

    return img



def crop_image1(img, tol=7):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return IDRID image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img






def scaleRadius(img,scale):

    k = img.shape[0]/2
    x = img[int(k), :, :].sum(1)
    r=(x>x.mean()/10).sum()/2
    if r == 0:
        r = 1
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)


def clahe(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(50, 50))
    #img = clahe.apply(img)
    return img

def grahams(img):
    sigmaX = 30
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return img


def pre_processings(img):

    #img = grahams(img)
    #img = clahe_green(img)
    img = clahe(img)


    return img

def run():


    # datasets = [f'datasets/messidor/messidor/messidor']
    #
    datasets = [
        [f'datasets/messidor_2/dataset'],
        [f'datasets/idrid/test'],
        [f'datasets/idrid/train'],
        [f'datasets/idrid/train'],
        [f'datasets/APTOS/test'],
        [f'datasets/APTOS/train'],
    ]



    for data in datasets:
        for path in glob.glob(data[0] + '/*'):
            name = path.split('/')
            final_name = name[-1:][0]
            #final_path = f'pre_processing/messidor/train/{final_name}'
            final_path = f'pre_processing/{data[0]}/{final_name}'

            img = cv2.imread(path)
            img = pre_processings(img)
            cv2.imwrite(final_path, img)
            print(final_path)



