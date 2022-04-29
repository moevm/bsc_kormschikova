import numpy as np
import cv2 as cv
from dataset_unpack import dataset_unpack
from glob import glob
import data_handler
import detect_border as dbe
import distance
import matplotlib.pyplot as plt
import master
import time
# import visualization


import io_ply_file

CAMERA_MTX = np.array([[1000.95935, 0.0, 799.6486],
                       [0.0, 1000.95935, 454.6965],
                       [0.0, 0.0, 1.0]])
MIN_MATCH_COUNT = 7
DEPTH_DATA_WRITE = True
SAVE_TMP_DATA = True
SAVE_CLOUD = True
FILE_3D = 'coordinates3D/n.txt'
FILE_PLY = 'coordinates3D/ne.ply'
VIDEO_NAME = '/distance_data/02'


def createExtrinsicParameters(rotationMat, translationMat):
    E = np.column_stack((rotationMat, translationMat))
    tmp = np.array([0,0,0,1])
    E = np.vstack((E, tmp))
    return E



def make4picVideo(arTop, arBot, dirname):
    new = []
    print('...save video')
    if(len(arTop) == len(arBot)):
        for i, n in enumerate(arTop):
            # print(i)
            # print(arTop[i].shape, arBot[i].shape)
            img = np.vstack((arTop[i], arBot[i]))

            new.append(img)
        h,w = img.shape[:2]
        dbe.videoWrite(new, (w,h), dirname+'vdistanceAndLines2')
    return


def testing(datasetDir):
    files = glob(datasetDir + '/images/*.png')
    files.sort()
    img1_ = files[0]
    img1 = cv.imread(img1_, 1)
    time_steps, true_distance, ground_truth_t, ground_truth_r = dataset_unpack(datasetDir)[:4]
    master.initAlgorithm(img1, FILE_3D, datasetDir)
    if SAVE_CLOUD:
        master.openFileToWrite3DCoord()
    border = False
    # time_start = time.time()

    for i, filename in enumerate(files[1:]):
        print(i)
        img2 = filename
        camera_move = ground_truth_t[i:i + 2]
        img2 = cv.imread(img2, 1)
        delta_t = time_steps[i+1] - time_steps[i]
        RFlag, LFlag, tmp, depIm = master.step_processing(img2, camera_move, ground_truth_r[i], ground_truth_t[i], delta_t, true_distance[i], border)
        # if RFlag:
        #     print('Rborder', i)
        # if LFlag:
        #     print('Lborder', i)
        border = RFlag or LFlag

        #only for save depth img TMP
        # fpath = datasetDir + '/distance/d'+str(i)+".png"
        # cv.imwrite(fpath, depIm)

        # img1 = img2
    print('main proc end ')
    data_handler.graphDistance(true_distance, master.getMeanDistance(), master.getFilterMeanDistance(), master.getMeanDistanceOnlyB(), datasetDir)
    # time_end = time.time()
    # print("testing new time", time_start - time_end)
    if SAVE_CLOUD:
        master.closeFileToWrite3DCoord()
    io_ply_file.create_point_cloud(FILE_3D, FILE_PLY)


def processing(datasetDir, ground_truth_t, ground_truth_r, true_distance):
    files = glob(datasetDir+'/images/*.png')
    files.sort()
    print(len(files))
    img1_ = files[0]
    img1 = cv.imread(img1_, 0)
    imgDepthAr = []
    # imgLineAr = []
    # coord3d = []
    distanceMean = []
    kp1, des1 = sift_algorithm(img1, SAVE_TMP_DATA, datasetDir + '/sift/_01' + '.png')

    for i, filename in enumerate(files[1:4]):
        print(i)
        img2 = filename
        name = img1_[-9:-4] + '-' + img2[-9:]
        camera_move = ground_truth_t[i:i+2]
        img2 = cv.imread(img2, 0)
        kp2, des2 = sift_algorithm(img2, SAVE_TMP_DATA, datasetDir+'/sift/'+name[-9:])
        mat = matches(des1, des2)
        saveMatches(mat,  img1, kp1, img2,kp2, datasetDir+'/matches/'+name[-9:])
        kpFirst, depth, img3 = distance.depth(kp1, kp2, mat, camera_move, img1)

        md = distance.distanceMeanWeight(depth, kp1, img1.shape[:2])
        distanceMean.append(md)
        # coord3dtmp = distance.coordinates_2d_to_3d(kpFirst, depth, img1.shape[:2], img1)[0]
        # coord3dtmp = distance.pointToLocalDroneСoordinates(coord3dtmp, ground_truth_r[i], ground_truth_t[i])
        # coord3d.extend(coord3dtmp)
        # RFlag, LFlag, RoofFlag, imgLine = dbe.detectBorder(kpFirst, depth,  img1)
        tmpStr = 'Count matches: '+str(len(mat))+' frame: ' + str(i+1)
        cv.putText(img3, tmpStr, (30, 830), cv.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 0),2, cv.LINE_AA)

        #
        # if RFlag:
        #     cv.putText(imgLine, "RIGHT BORDER!!!", (450, 500), cv.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 4, cv.LINE_AA)
        # if LFlag:
        #     cv.putText(imgLine, "LEFT BORDER!!!", (450, 700), cv.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 4, cv.LINE_AA)
        # cv.imshow('a', img3)
        # # cv.imshow('a+', img1)
        # # cv.imshow('b', imgLine)
        # cv.waitKey(0)
        # print(img3.shape)
        # imgLineAr.append(imgLine)
        imgDepthAr.append(img3)
        img1 = img2
        img1_ = filename
        kp1 = kp2
        des1 = des2

    print('proc end')
    # print(len(distanceMean), len(true_distance))

    # visualization.plot3Dcoord(coord3d)
    # print(imgDepthAr[0].shape, imgLineAr[0].shape)
    # make4picVideo(imgDepthAr, imgLineAr, datasetDir+VIDEO_NAME)
    # height, width = imgLine.shape[:2]
    # dbe.videoWrite(imgLineAr,  (width, height))
    height, width = img1.shape
    data_handler.videoWrite(imgDepthAr,  (width, height), datasetDir+'/distance_data/1')


    print(distanceMean)
    true_distance = true_distance/100
    # print(true_distance)

    px = 1 / plt.rcParams['figure.dpi']
    plt.subplots(figsize=(1600 * px, 600 * px))
    plt.grid()
    y2 = list(true_distance[:len(distanceMean)])
    print(len(y2))
    maxD = 35
    data_handler.writeDistanseData([distanceMean], datasetDir + '/distance_data/distanse_mean_1.txt')
    for i, a in enumerate(distanceMean):
        if a > maxD:
            distanceMean[i] = maxD
        elif a < 0:
            distanceMean[i] = -1
        if(i < len(y2)):
            if(y2[i] > maxD):
                y2[i] = maxD
        else:
            print(i)
            y2.extend([-1])
    # distanceMean = list(filter(lambda x: x < 20, distanceMean))
    x = [a for a in range(len(distanceMean))]
    plt.plot(x, distanceMean, label='exp')
    plt.plot(x, y2, label='true')  # Plot more data on the axes...
    plt.xlabel('iteration')
    plt.ylabel('distance, m')
    plt.legend()
    plt.savefig(datasetDir+'/distance_data/meanDistance+true_6.png')

    if (DEPTH_DATA_WRITE):
        data_handler.writeDistanseData(distance.getDepthData(), datasetDir+'/distance_data/distanse_3.txt')
        data_handler.writeDistanseData([distanceMean], datasetDir+'/distance_data/distanse_mean_3.txt')

    # data_handler.GraphDistance(datasetDir+'/distance_data/distanse_3.txt')
    # data_handler.rotationGraph(ground_truth_r, datasetDir)



def sift_algorithm(img, save=False, filepath='s.png'):
    # img = cv.imread(img)
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    if save:
        # imgname = filepath + img[-9:]
        img = cv.drawKeypoints(img, kp, img)
        cv.imwrite(filepath, img)
    return kp, des






FILE_3D = 'coordinates3D/dt_old26-04.txt'
FILE_PLY = 'coordinates3D/dt_old26-04.ply' #
dataset = 'datasets/dataset_06_new'
testing(dataset)

# io_ply_file.create_point_cloud(FILE_3D, FILE_PLY)

# FILE_PLY = 'coordinates3D/tmp_start_d2_offset.ply' #
exit(0)
#TODO: add to dataset unpack
offset_tmp = np.genfromtxt(dataset + "/" + "start_trajectory.txt", dtype=str)
print(offset_tmp)

offset = [float(offset_tmp[0][2:])/100,
        float(offset_tmp[1][2:])/100,
        float(offset_tmp[2][2:])/100]

# io_ply_file.create_point_cloud(FILE_3D, FILE_PLY, offset )


#
# FILE_3D = 'coordinates3D/dist_er_d.txt'
# FILE_PLY = 'coordinates3D/dist_er_d.ply'
# dataset = 'datasets/new_datasets/und/dt2'
# testing(dataset)



# time_steps, true_distance, ground_truth_t, ground_truth_r = dataset_unpack(dataset)[:4]
# processing(dataset, ground_truth_t, ground_truth_r, true_distance)
# # #
# dataset = 'datasets/datasets_roof/roof02'
# time_steps, true_distance, ground_truth_t, ground_truth_r = dataset_unpack(dataset)[:4]
# processing(dataset, ground_truth_t, ground_truth_r, true_distance)
#
# dataset = 'datasets/datasets_roof/roof03'
# time_steps, true_distance, ground_truth_t, ground_truth_r = dataset_unpack(dataset)[:4]
# processing(dataset, ground_truth_t, ground_truth_r, true_distance)
#


# data_handler.GraphDistance('dataset_02/distance_data/distance_data.txt')
# data_handler.GraphDistance('dataset_05/distance_data/distance_data_2.txt')
# data_handler.GraphDistance('dataset06_ud/distance_data/distance_data_2.txt')
# time_steps, ground_truth_t, ground_truth_r = dataset_unpack('dataset_02')
# data_handler.GraphDistance()
# data_handler.rotationGraph(ground_truth_r, 'dataset_02/distance_data')
# true_distance /= 100
# processing('dataset_06_new', ground_truth_t, ground_truth_r, true_distance)
# points = []
# points.append([0,0,1])
# points.append([0,2,1])
# points.append([0,3,1])
#
# pointToLocalDroneСoordinates(points, ground_truth_r[0], ground_truth_t[0])

# print(true_distance)

# data_handler.GraphDistanceTrue(true_distance, 'dataset_06_new'+'/distance_data/m_distance_data_2.txt')

# data_handler.rotationGraph(ground_truth_r, 'dataset_03/distance_data')
# time_steps, ground_truth_t, ground_truth_r = dataset_unpack('dataset_04')
# data_handler.rotationGraph(ground_truth_r, 'dataset_04/distance_data')

#
# def testLines(datasetDir):
#     files = glob(datasetDir + '/images/*.png')
#     files.sort()
#     img1_ = files[0]
#     img1 = cv.imread(img1_, 0)
#     imgAr = []
#
#     for i, filename in enumerate(files[1:3]):
#         img2 = filename
#         name = img1_[-9:-4] + '-' + img2[-9:]
#         img2 = cv.imread(img2, 0)
#         # kp1, des1 = sift_algorithm(img1, SAVE_TMP_DATA, datasetDir+'/sift/'+name[:5]+'.png')
#         # kp2, des2 = sift_algorithm(img2, SAVE_TMP_DATA, datasetDir+'/sift/'+name[-9:])
#         # mat = matches(des1, des2)
#         re = dbe.iWallToDie(img1, [])
#         imgAr.append(re)
#         img1 = img2
#         img1_ = filename
#         print(i)
#     print('proc end')
#     height, width = re.shape[:2]
#     # dbe.videoWrite(imgAr,  (width, height))
#
#     return
#
# def testing(matches, img1, img2, kp1, kp2, name ):
#     # epilines
#     pts1 = []
#     pts2 = []
#     # ratio test as per Lowe's paper
#     for i, (m, n) in enumerate(matches):
#         pts2.append(kp2[m.trainIdx].pt)
#         pts1.append(kp1[m.queryIdx].pt)
#     pts1 = np.int32(pts1)
#     pts2 = np.int32(pts2)
#
#     F, maskF = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
#     # Find epilines corresponding to points in right image (second image) and
#     # drawing its lines on left image
#     lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
#     lines1 = lines1.reshape(-1, 3)
#     img5, img6 = visualization.drawlines(img1, img2, lines1, pts1, pts2)
#     # Find epilines corresponding to points in left image (first image) and
#     # drawing its lines on right image
#     lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
#     lines2 = lines2.reshape(-1, 3)
#     img3, img4 = visualization.drawlines(img2, img1, lines2, pts2, pts1)
#     vis = np.concatenate((img5, img3), axis=1)
#     cv.imwrite('dataset_02/epilines/'+name, vis)
#     #endEpilines
#     return

#

def matches(firstDes, secondDes):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(firstDes, secondDes, k=2)
    goodMatches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            goodMatches.append(matches[i])
    return goodMatches


def saveMatches(matches, img1, kp1, img2, kp2, filepath='matches.jpg'):
    matchesMask = [[1, 0] for i in range(len(matches))]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask[:],
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img_3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches[:], None, **draw_params)
    cv.imwrite(filepath, img_3)
