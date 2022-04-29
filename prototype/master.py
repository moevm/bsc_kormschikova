import numpy as np
import cv2 as cv

import detect_border
import distance
import io_ply_file
import sift_and_matches
import Kalman
from constant_parameters import MAX_DRON_DISTANCE as MAX_DISTANCE
import pyransac3d as ransac

WRITE_3D_COORD = False
DIR = ''
FILEPATH_FOR_3D_COORD = 'coordinates.txt'
FILE = None
KP_FIRST_IMG = None
DES_FIRST_IMG = None
IMG_F_BGR = None
IMG_F_GRAY = None
MEAN_DISTANCES = []
PREV_MEAN = -1
MEAN_DISTANCES_ONLY_B = []
FILTER_MEAN_DISTANCE = []
DELTA_DISTANCE_POINTS = 3
DELTA_DISTANCE_BUILDINGS = 1
# MAX_DISTANCE = 25
DISTANCE_BUILD = 0
STEP = 0
COORD_TMP = []
COLOR_TMP = []


##TODO: true dist

def step_processing(img_second_bgr, camera_move, ground_truth_r, ground_truth_t, delta_t, true_dist = ' DELETE THIS', border=False):
    global KP_FIRST_IMG, DES_FIRST_IMG, IMG_F_GRAY, IMG_F_BGR, DISTANCE_BUILD, PREV_MEAN, STEP
    img_first = IMG_F_GRAY
    img_second = cv.cvtColor(img_second_bgr, code=cv.COLOR_BGR2GRAY);
    kp1 = KP_FIRST_IMG
    des1 = DES_FIRST_IMG
    kp2, des2 = sift_and_matches.sift_algorithm(img_second)
    matches = sift_and_matches.detect_matches(des1, des2)
    kpFirstGood, depth, TMPIMG = distance.depth(kp1, kp2, matches, camera_move, img_first)#[:2]
    mean_distance = distance.distanceMeanWeight(depth, kp1, img_first.shape[:2]) #only for grapgh
    coord, color = distance.coordinates_2d_to_3d(kpFirstGood, depth, img_first.shape[:2], IMG_F_BGR)
    RFlag, LFlag, RoofFlag, TMP_IMG_2 = detect_border.detectBorder(kpFirstGood, depth, img_first)#[:2]

   # mean_distance_for_filter = distance.distanceMean3DTesting(coord, MAX_DISTANCE)

    if PREV_MEAN == -1:
        prev_distance = mean_distance #mdff
        DISTANCE_BUILD = mean_distance #mdff

    else:
        prev_distance = PREV_MEAN
    # print('DISTANCE B ', DISTANCE_BUILD)

    MEAN_DISTANCES.append(mean_distance)#only for graph
    ##TODO: KALMAN FILTER FIX
    filter_distance = Kalman.step_kalman(delta_t, mean_distance, prev_distance)#mdff
    # print('filter ', filter_distance)
    # if RFlag or LFlag:
    #     if RFlag:
    #         depth_r, depth_overseas_r = detect_border.removeDistanceBeyondBorder(depth, kpFirstGood, 'r')
    #         # print('RFLAG DIST', np.mean(depth_r), np.mean(depth_overseas_r))
    #         # if DISTANCE_BUILD == MAX_DISTANCE: #means there were no build
    #         #     print('from nothing')
    #         #
    #         #     filter_distance = np.mean(depth_r)
    #         #     DISTANCE_BUILD = np.mean(depth_r) + DELTA_DISTANCE_BUILDINGS + 1
    #         #
    #         # else:
    #         filter_distance = np.mean(depth_overseas_r)
    #
    #     if LFlag:
    #         depth_l, depth_overseas_l = detect_border.removeDistanceBeyondBorder(depth, kpFirstGood, 'l')
    #         # print('LFLAG DIST', np.mean(depth_l), np.mean(depth_overseas_l))
    #         # if DISTANCE_BUILD == MAX_DISTANCE:
    #         #     print('from nothing')
    #         #     filter_distance = np.mean(depth_l)
    #         #     DISTANCE_BUILD = np.mean(depth_l) + DELTA_DISTANCE_BUILDINGS + 1
    #         # else:
    #         filter_distance = np.mean(depth_overseas_l)
    #
    #     if np.abs(DISTANCE_BUILD - filter_distance) >= DELTA_DISTANCE_BUILDINGS:
    #         print('newBilding', prev_distance, filter_distance)
    #         DISTANCE_BUILD = filter_distance
    #         Kalman.reset_kalman(filter_distance)
    #         MEAN_DISTANCES.pop()
    #         MEAN_DISTANCES.append(filter_distance)
    #         # prev_distance
    #         # cv.imshow('noB', TMPIMG)
    #         # cv.waitKey(0)
    #     else:
    #         print('FALSE BORDER')
    #         RFlag = False
    #         LFlag = False
    #         DISTANCE_BUILD = MAX_DISTANCE

    # coord, color = distance.removeWrongCoord(coord, color, filter_distance + DELTA_DISTANCE_POINTS,
    #                                          filter_distance - DELTA_DISTANCE_POINTS)

    # coord, color = distance.removeWrongCoord(coord, color, MAX_DISTANCE, 3)
    # print(len(coord))
    # coord, color = testRANSAC(coord, color)

    #
    # if filter_distance >= MAX_DISTANCE:
    #     print('thereIsNoBuilding', filter_distance, DISTANCE_BUILD)
    #     DISTANCE_BUILD = MAX_DISTANCE
    #     Kalman.reset_kalman(MAX_DISTANCE)
    #     MEAN_DISTANCES.pop()
    #     MEAN_DISTANCES.append(MAX_DISTANCE)
    #     filter_distance = MAX_DISTANCE
    #     coord = []
    #     color = []
    #     RFlag = False #???
    #     LFlag = False
    # else:
    #     print('build')

    PREV_MEAN = filter_distance
    FILTER_MEAN_DISTANCE.append(filter_distance)
    KP_FIRST_IMG = kp2
    DES_FIRST_IMG = des2
    IMG_F_GRAY = img_second
    IMG_F_BGR = img_second_bgr

    if WRITE_3D_COORD:
        write_3d_to_file(coord, ground_truth_r, ground_truth_t, color)
        #STEP PROCESSING
        # global COORD_TMP, COLOR_TMP
        # COORD_TMP.extend(coord)
        # COLOR_TMP.extend(color)
        # print("coord-color",len(COORD_TMP), len(COLOR_TMP))
        # stepForLayer = 6
        # if STEP% stepForLayer == 0:
        #     write_3d_to_file_for_step_and_ply(COORD_TMP, ground_truth_r, ground_truth_t, COLOR_TMP, DIR+'/step_cloud/s'+str(STEP)+'.txt',DIR+'/step_cloud/s_'+str(STEP)+'.ply' )
        #     COORD_TMP.clear()
        #     COLOR_TMP.clear()
    # STEP+=1

    saveTruthAndKalmanOnePoint(true_dist, filter_distance, ground_truth_t)
    return RFlag, LFlag, filter_distance, TMPIMG


def write_3d_to_file(coord, ground_truth_r, ground_truth_t, color):
    global FILE
    coord_2 = distance.pointToLocalDroneСoordinates(coord, ground_truth_r, ground_truth_t)
    if FILE is not None:
        if FILE.closed == False:
            for ind, i in enumerate(coord_2):
                tmpStr = str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(color[ind][2]) + ' ' + str(color[ind][1]) + ' ' + str(color[ind][0]) + ' '
                FILE.write(tmpStr)
            FILE.write(';\n')

def write_3d_to_file_for_step_and_ply(coord, ground_truth_r, ground_truth_t, color, namefile, ply):
    if(len(coord) == 0):
        return
    coord = distance.pointToLocalDroneСoordinates(coord, ground_truth_r, ground_truth_t)
    f = open(namefile, 'w')
    if f is not None:
        if f.closed == False:
            for ind, i in enumerate(coord):
                tmpStr = str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(color[ind][2]) + ' ' + str(
                    color[ind][1]) + ' ' + str(color[ind][0]) + ' '
                f.write(tmpStr)
            f.write(';\n')
    f.close()
    io_ply_file.create_point_cloud(namefile, ply)

def initAlgorithm(img_first,  filepath3d = '', dir =''):
    global KP_FIRST_IMG, DES_FIRST_IMG, IMG_F_GRAY, IMG_F_BGR, FILEPATH_FOR_3D_COORD, STEP, DIR, MEAN_DISTANCES, MEAN_DISTANCES_ONLY_B, FILTER_MEAN_DISTANCE, PREV_MEAN
    MEAN_DISTANCES.clear()
    PREV_MEAN = -1
    MEAN_DISTANCES_ONLY_B.clear()
    FILTER_MEAN_DISTANCE.clear()
    STEP = 0
    IMG_F_BGR = img_first
    if dir != '':
        DIR = dir
    if filepath3d != '':
        FILEPATH_FOR_3D_COORD = filepath3d
    IMG_F_GRAY = cv.cvtColor(img_first, code=cv.COLOR_BGR2GRAY);
    KP_FIRST_IMG, DES_FIRST_IMG = sift_and_matches.sift_algorithm(IMG_F_GRAY)


def openFileToWrite3DCoord():
    global WRITE_3D_COORD, FILE
    WRITE_3D_COORD = True
    FILE = open(FILEPATH_FOR_3D_COORD, 'w')
    return

def saveTruthAndKalmanOnePoint(true_distance, kalman_distanse, ground_truth_t):
    colorTrue = [0,255,0]
    colorKalman = [0,0,255]
    ground_truth_t = ground_truth_t/100
    pointKalman = [ground_truth_t[0],ground_truth_t[1]+kalman_distanse, ground_truth_t[2]]  #ONLY FOR ROOF 06
    pointTrue =  [ground_truth_t[0],ground_truth_t[1]+true_distance/100, ground_truth_t[2]]
    coord = [pointTrue, pointKalman]
    color = [colorTrue, colorKalman]
    # print( pointKalman, pointTrue, ground_truth_t, sep='\n' )
    if FILE is not None:
        if FILE.closed == False:
            for ind, i in enumerate(coord):
                tmpStr = str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(color[ind][2]) + ' ' + str(color[ind][1]) + ' ' + str(color[ind][0]) + ' '
                FILE.write(tmpStr)
            FILE.write(';\n')



def closeFileToWrite3DCoord():
    global WRITE_3D_COORD, FILE
    WRITE_3D_COORD = False
    FILE.close()
    FILE = None
    return

def testRANSAC(coord, color):

    if len(coord) < 4:
        return coord, color
    newCoord = []
    newColor = []
    plane1 = ransac.Plane()
    best_eq, best_inliers = plane1.fit(np.array(coord), 1)
    for i in best_inliers:
        newCoord.append(coord[i])
        newColor.append(color[i])
    return newCoord, newColor


def getMeanDistance():
    return MEAN_DISTANCES


def getFilterMeanDistance():
    return FILTER_MEAN_DISTANCE

def getMeanDistanceOnlyB():
    return MEAN_DISTANCES_ONLY_B