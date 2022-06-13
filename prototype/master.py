import numpy as np
import cv2 as cv
from glob import glob
from dataset_unpack import dataset_unpack
import detect_border
import distance
import data_handler
import io_ply_file
import sift_and_matches
import Kalman
from constant_parameters import MAX_DRON_DISTANCE as MAX_DISTANCE
import pyransac3d as ransac
import time


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
TMP_TIME_PLY = [0]

##TODO: true dist


def dataset_processing(dataset_dir, file3d_name, filePLY_name, depth_img_flag):
    files = glob(dataset_dir + '/images/*.png')
    files.sort()
    img1_ = files[0]
    img1 = cv.imread(img1_, 1)
    time_steps, true_distance, ground_truth_t, ground_truth_r = dataset_unpack(dataset_dir)[:4]
    initAlgorithm(img1, file3d_name, dataset_dir)
    border = False
    time_start = time.time()
    lenf = len(files)
    print('start: ', dataset_dir)
    for i, filename in enumerate(files[1:]):
        print('img: ', i, '/', lenf)
        img2 = filename
        camera_move = ground_truth_t[i:i + 2]
        img2 = cv.imread(img2, 1)
        delta_t = time_steps[i + 1] - time_steps[i]

        RFlag, LFlag, tmp, depIm = step_processing(img2, camera_move, ground_truth_r[i], delta_t, true_distance[i])
        if(depth_img_flag):
            fpath = dataset_dir + '/distance/d_' + str(i) + ".png"
            cv.imwrite(fpath, depIm)
    print('main proc end ')
    time_end = time.time()
    print("testing  time", time_end - time_start)

    data_handler.graphDistance(true_distance, getMeanDistance(), getFilterMeanDistance(),
                               getMeanDistanceOnlyB(), dataset_dir)
    closeFileToWrite3DCoord()
    print('create points')
    # write_3d_to_file_for_step_and_ply(COORD_TMP, COLOR_TMP, dataset_dir + '/step_cloud/new_step_end_remove.txt',
    #                                   dataset_dir + '/step_cloud/new_step_end_remove.ply')
    #
    io_ply_file.create_point_cloud(file3d_name, filePLY_name)
    # print("testing  time", time_end - time_start)

    return

def step_processing(img_second_bgr, camera_move, ground_truth_r,  delta_t = 0, true_dist = 0, border=False):
    """
    Данная функция проводит одну итерацию метода по построению облака точек, подходит для работы в режиме "онлайн"
    Перед первым вызовом необходимо инициализировать алгоритм с помощью initAlgorithm(), затем openFileToWrite3DCoord()


    :param img_second_bgr:  изображение в формате BGR, которое будет обрабатываться и сравниваться с предыдущим
    :param camera_move: координата дрона на предыдущем и в этом кадре
    в формате [[x_first, y_first, z_first],[x_second, y_second, z_second]]
    :param ground_truth_r: вращение дрона в формате [pitch, yaw, roll]
    :param delta_t: время прошедшее между кадрами
    :param border: флаг границы здания
    :return: RFlag, LFlag - флаги границ здания filter_distance - средневзвешенная дистанция
    TMPIMG - изображения с контрольными точками и дистанциями до них
    """
    global KP_FIRST_IMG, DES_FIRST_IMG, IMG_F_GRAY, IMG_F_BGR, DISTANCE_BUILD, PREV_MEAN, STEP
    ground_truth_t = camera_move[0]
    img_first = IMG_F_GRAY
    img_second = cv.cvtColor(img_second_bgr, code=cv.COLOR_BGR2GRAY);
    kp1 = KP_FIRST_IMG
    des1 = DES_FIRST_IMG
    kp2, des2 = sift_and_matches.sift_algorithm(img_second, False, DIR+'/sift/'+str(STEP) +'.png')
    matches = sift_and_matches.detect_matches(des1, des2)
    # sift_and_matches.save_matches_img(matches, img_first, kp1, img_second, kp2, DIR+'/matches/'+str(STEP) +'.png')
    kpFirstGood, depth, TMPIMG = distance.depth(kp1, kp2, matches, camera_move, img_first)#[:2]
    mean_distance = distance.distanceMeanWeight(depth, kp1, img_first.shape[:2]) #only for grapgh
    coord, color = distance.coordinates_2d_to_3d_old(kpFirstGood, depth, img_first.shape[:2], IMG_F_BGR)
    RFlag, LFlag, RoofFlag, TMP_IMG_2 = detect_border.detectBorder(kpFirstGood, depth, img_first)#[:2]

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
    if filter_distance >= MAX_DISTANCE:
        # print('thereIsNoBuilding', filter_distance, DISTANCE_BUILD)
        DISTANCE_BUILD = MAX_DISTANCE
        Kalman.reset_kalman(MAX_DISTANCE)
        # MEAN_DISTANCES.pop()
        # MEAN_DISTANCES.append(MAX_DISTANCE)
        filter_distance = MAX_DISTANCE
        coord = []
        color = []
        RFlag = False #???
        LFlag = False


    PREV_MEAN = filter_distance
    FILTER_MEAN_DISTANCE.append(filter_distance)
    KP_FIRST_IMG = kp2
    DES_FIRST_IMG = des2
    IMG_F_GRAY = img_second
    IMG_F_BGR = img_second_bgr

    if WRITE_3D_COORD:
        write_3d_to_file(coord, ground_truth_r, ground_truth_t, color)
        # STEP PROCESSING
        # global COORD_TMP, COLOR_TMP
        # coord = distance.pointToLocalDroneСoordinates(coord, ground_truth_r, ground_truth_t)
        # COORD_TMP.extend(coord)
        # COLOR_TMP.extend(color)
        # stepForLayer = 250
        # if STEP% stepForLayer == 0:
        #     print(STEP)
        #     write_3d_to_file_for_step_and_ply(COORD_TMP, COLOR_TMP, DIR+'/step_cloud/new_step_remove'+str(STEP)+'.txt',DIR+'/step_cloud/new_step_remove_'+str(STEP)+'.ply' )
        #     COORD_TMP.clear()
        #     COLOR_TMP.clear()
    STEP+=1

    # saveTruthAndKalmanOnePoint(true_dist, filter_distance, ground_truth_t)
    return RFlag, LFlag, filter_distance, TMPIMG

def write_3d_to_file(coord, ground_truth_r, ground_truth_t, color):
    global FILE
    coord_2 = distance.pointsFromLocalDroneСoordinates(coord, ground_truth_r, ground_truth_t)
    if FILE is not None:
        if FILE.closed == False:
            for ind, i in enumerate(coord_2):
                tmpStr = str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(color[ind][2]) + ' ' + str(color[ind][1]) + ' ' + str(color[ind][0]) + ' '
                FILE.write(tmpStr)
            FILE.write(';\n')

def write_3d_to_file_for_step_and_ply(coord, color, namefile, ply):
    if(len(coord) == 0):
        return
    f = open(namefile, 'w')
    if f is not None:
        if f.closed == False:
            for ind, i in enumerate(coord):
                tmpStr = str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + ' ' + str(color[ind][2]) + ' ' + str(
                    color[ind][1]) + ' ' + str(color[ind][0]) + ' '
                f.write(tmpStr)
            f.write(';\n')
    f.close()
    print('create points')
    time_start_ply = time.time()
    io_ply_file.create_point_cloud(namefile, ply)
    time_end_ply = time.time()
    tmp =  time_end_ply - time_start_ply
    print("PLY TIME:", tmp)
    TMP_TIME_PLY.append(tmp)

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
    openFileToWrite3DCoord()

def openFileToWrite3DCoord():
    global WRITE_3D_COORD, FILE
    WRITE_3D_COORD = True
    FILE = open(FILEPATH_FOR_3D_COORD, 'w')
    return

def saveTruthAndKalmanOnePoint(true_distance, kalman_distanse, ground_truth_t):
    colorTrue = [0,255,0]
    colorKalman = [0,0,255]
    ground_truth_t = ground_truth_t/100
    # pointKalman = [ground_truth_t[0],ground_truth_t[1]+kalman_distanse, ground_truth_t[2]]  #ONLY FOR ROOF 06
    point_ground_t = [ground_truth_t[0],ground_truth_t[1], ground_truth_t[2]]  #ONLY FOR ROOF 06

    pointTrue = [ground_truth_t[0]+true_distance/100,ground_truth_t[1], ground_truth_t[2]]
    coord = [pointTrue, point_ground_t]
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