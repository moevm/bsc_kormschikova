import numpy as np
import cv2 as cv
from constant_parameters import HORIZONTAL_FOV, VERTICAL_FOV, MAX_DRON_DISTANCE, MIN_DRON_DISTANCE, CAMERA_MTX

# CAMERA_MTX = np.array([[1000.95935, 0.0, 799.6486],
#                        [0.0, 1000.95935, 454.6965],
#                        [0.0, 0.0, 1.0]])
MIN_MATCH_COUNT = 7
DEPTH_DATA_WRITE = True
DEPTH_TO_IMG = True
DEPTH_DATA = []


def getDepthData():
    return DEPTH_DATA

def depth(kpFirst, kpSecond, goodMatches, cameraMove, img=None):
    kpFirstGood = []
    kpSecondGood = []
    depth = []
    dist = 0

    if (len(cameraMove) > 1):  # tmp
        dist = np.sqrt((cameraMove[0][0] - cameraMove[1][0]) ** 2 + (cameraMove[0][1] - cameraMove[1][1]) ** 2 + (
                cameraMove[0][2] - cameraMove[1][2]) ** 2)
    f = CAMERA_MTX[0][0]
    tmpStr = 'Move: ' + str(dist)[:4]

    if len(goodMatches) == 0 or dist == 0:
        depth.append(0)
        if DEPTH_DATA_WRITE:
            DEPTH_DATA.append(depth)
            # print(DEPTH_DATA)
        if DEPTH_TO_IMG and img is not None:
            tmpImg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.putText(tmpImg, '-1', (30, 850), cv.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 0), 2, cv.LINE_AA)
            # print('len dist = 0, sh', tmpImg.shape)
        return kpFirstGood, depth, tmpImg
    for i, match in enumerate(goodMatches):
        tmp_f = kpFirst[match[0].queryIdx]
        tmp_s = kpSecond[match[0].trainIdx]

        difX = np.sqrt(
            (tmp_f.pt[0] - tmp_s.pt[0]) ** 2 + (tmp_f.pt[1] - tmp_s.pt[1]) ** 2)

        z = f * dist / difX  #cm
        # if MIN_DRON_DISTANCE < z:
        #     depth.append(z / 100)
        #     kpFirstGood.append(kpFirst[match[0].queryIdx])
        #     kpSecondGood.append(kpSecond[match[0].trainIdx])
        depth.append(z / 100) # m
        kpFirstGood.append(kpFirst[match[0].queryIdx])
        kpSecondGood.append(kpSecond[match[0].trainIdx])
        # else:
        #     print('remove dist', z)

    # z = (fT)/d
    if DEPTH_DATA_WRITE:
        DEPTH_DATA.append(depth)
    if DEPTH_TO_IMG and img is not None:
        tmpImg = drawDepthCercle(kpFirstGood, depth, img)
        cv.putText(tmpImg, tmpStr, (30, 850), cv.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 0), 2, cv.LINE_AA)
        # cv.imshow('testingDepth', tmpImg)
        # cv.waitKey(0)
        # print('len dist norm, sh', tmpImg.shape)
    return kpFirstGood, depth, tmpImg


def distanceMean3DTesting(coord, max):
    if (len(coord) != 0):
        # print('3d coord:', len(coord))
        l = 0
        for c in coord:
            # print(c[2], c)
            l += c[2]
        # print("--------", l / len(coord))
        res = l / len(coord)
    else:
        res = max
    return res


# def coordinates_2d_to_3d(kp, depth, imgshape, img):  # testing! not finished
#     if len(depth) == 0:
#         return [[0, 0, -1]], [0, 0, 0]
#     if depth[0] == -1:
#         return [[0, 0, -1]], [0, 0, 0]
#     pointCoord = []
#     color = []
#     f = CAMERA_MTX[0][0] / 100 #m
#     img_x_m = 2.78683 * f / 2 #m
#     img_y_m = 1.94073 * f / 2 #m
#     for i in range(len(kp)):
#         Cx = [kp[i].pt[0] - imgshape[1] / 2, #pixel
#               -kp[i].pt[1] + imgshape[0] / 2, #pixel
#               f] #m
#         len_px = (Cx[0] / (imgshape[1] / 2)) * img_x_m #m
#         len_py = (Cx[1] / (imgshape[0] / 2)) * img_y_m #m
#         lenPX = np.sqrt(len_px ** 2 + len_py ** 2) #m
#         lenCX = np.sqrt(f ** 2 + lenPX ** 2) #m вектор от камеры до точки на "экране"
#         Cx[0] = len_px
#         Cx[1] = len_py
#         # Cx[2] = f
#         # koef = depth[i] / lenCX
#         # tmp =  np.dot(koef, Cx)
#         CX = np.dot(depth[i] / lenCX, Cx) #m
#         pointCoord.append(list(CX))
#         color.append(img[int(kp[i].pt[1])][int(kp[i].pt[0])])
#     return pointCoord, color
#

def coordinates_2d_to_3d_old(kp, depth, imgshape, img):
    if (len(depth) == 0):
        return [[0, 0, -1]], [0, 0, 0], depth
    if(depth[0] == -1):
        return [[0, 0, -1]], [0,0,0], depth
    # print(depth)
    pointCoord = []
    color = []
    f = CAMERA_MTX[0][0]/100
    for i in range(len(kp)):
        Cx = [kp[i].pt[0] - imgshape[1]/2,#pixel
              -kp[i].pt[1] + imgshape[0]/2,#pixel
              f]   #m
        lenpxX = (Cx[0]/(imgshape[1]/2))*f #m

        angle = 90 - VERTICAL_FOV/2
        lenpy = f * np.sin(np.pi/180*(VERTICAL_FOV/2))/np.sin(np.pi/180*angle)#m  половина стороны изображения
        lenpxY = (Cx[1] / (imgshape[0]/2)) * lenpy#m

        lenp_x = np.sqrt(lenpxX**2 + lenpxY**2)#m
        lenCX = np.sqrt(f**2+lenp_x**2) #m вектор от камеры до точки на "экране"
        Cx[0] = lenpxX
        Cx[1] = lenpxY
        Cx[2] = lenCX #or f idk
        CX = np.dot(depth[i]/lenCX, Cx)
        # depth[i] = CX[2]
        # if(CX[2] <= maxDepth):
        pointCoord.append(list(CX))
        color.append(img[int(kp[i].pt[1])][int(kp[i].pt[0])])
    return pointCoord, color#, depth


def removeWrongCoord(coord, color, maxDepth, minDepth=0):
    newCoord = []
    newColor = []
    for i, c in enumerate(coord):
        if c[2] <= maxDepth and c[2] >= minDepth:
            # print(c, c[2])
            newCoord.append(c)
            newColor.append(color[i])
    # print(len(coord), len(newCoord))
    return newCoord, newColor


def distanceMeanWeight(depth, kp, imgShape):
    if (len(depth) <= 1):
        return 0
    maxLen = np.sqrt(imgShape[1] ** 2 + imgShape[0] ** 2)
    dictionary = dict(zip(depth, kp))
    sigma = np.std(depth)
    mean = np.mean(depth)
    depthList = list(filter(lambda x: sigma - mean <= x[0] <= mean + sigma, dictionary.items()))
    # depthList = list(filter(lambda x: MIN_DRON_DISTANCE <= x[0] <= MAX_DRON_DISTANCE, dictionary.items()))

    meanDist = 0
    sumWeight = 0
    for i in depthList:
        fromCenter = (i[1].pt[0] - imgShape[1] / 2, - i[1].pt[1] + imgShape[0] / 2)
        weight = 1 - np.sqrt(fromCenter[0] ** 2 + fromCenter[1] ** 2) / maxLen
        sumWeight += weight
        meanDist += i[0] * weight
    if (sumWeight != 0):
        meanDist /= sumWeight
    else:
        for i in depthList:
            fromCenter = (i[1].pt[0] - imgShape[1] / 2, - i[1].pt[1] + imgShape[0] / 2)
            weight = 1 - np.sqrt(fromCenter[0] ** 2 + fromCenter[1] ** 2) / maxLen
            sumWeight += weight
            meanDist += i[0] * weight
        meanDist = -1

    return meanDist


def drawDepthCercle(kp_pt, depth, img):
    if (len(depth) == 0):
        return img
    depthTMP = depth
    depth = depth / max(depth)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    for i, pt in enumerate(kp_pt):

        red = 255 - depth[i] * 255
        blue = 0
        green = 0
        if (red < 45):
            blue = np.exp(depth[i] * 5)
            red = 27
        if (red > 150):
            green = 255 - np.exp(depth[i] * 6) * 20
            red = 240
        color = (blue, green, red)  # bgr
        point = (int(pt.pt[0]), int(pt.pt[1]))
        img = cv.circle(img, point, 3, color, -1)
        if i % 3 != 0:
            continue
        tmpStr = str(depthTMP[i])[:4] + "m."
        cv.putText(img, tmpStr, (point[0] + 5, point[1] - 25), cv.FONT_HERSHEY_PLAIN, 1, (80, 0, 255), 1, cv.LINE_AA)

    return img


def createRotationMatrix(pitch, yaw, roll):
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    roll = np.deg2rad(roll)
    Ryaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [np.sin(yaw), np.cos(yaw), 0],
                     [0, 0, 1]])  # z
    Rpitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])  # y
    Rroll = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])  # x
    R = np.dot(np.dot(Ryaw, Rpitch), Rroll)
    return R

def pointsFromLocalDroneСoordinates(points, rotation, translation):
    newPoints = []
    R = createRotationMatrix(rotation[0] * -1, rotation[1] * -1, rotation[2] * -1)
    for point in points:
        tmp = np.array([point[2],  # (0, 0, 1)
                        point[0],  # (1, 0, 0)
                       point[1]])  # (0, 1, 0)
        # tmp = np.array([point[0],  # (1, 0, 0)
        #                 point[1],  # (0, 1, 0)
        #                 point[2]])  # (0, 0, 1)
        # tmp = np.dot(R, tmp) #m
        # tmp = np.array([tmp[1],  # (0, 1, 0)
        #                 tmp[2],  # (0, 0, 1)
        #                 tmp[0]])  # (1, 0, 0)
        tmp = np.dot(R, tmp) #m

        tmp = tmp + (translation / 100) #translation - m
        newPoints.append(list(tmp))
    return newPoints
