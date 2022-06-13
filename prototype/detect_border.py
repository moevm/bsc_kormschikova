import numpy as np
import cv2 as cv
from constant_parameters import MAX_DRON_DISTANCE
color_base = (0, 100, 255)
horizontal_color = (100, 0, 100)
vertical_color = (100, 255, 0)
colorLB = (100, 255, 255)  # yellow
colorRB = (255, 155, 100)  # blue
colorRoof = (255, 150, 255) #pink

MIN_DEPTH_LEN = 8
PIXEL_DX = 200  # расстояние в пикселях, на основании которого точки распределяются в левую и в правую стороны
PIXEL_CENTER_DX = 50   # расстояние в пикселях, на основании которого выбираются еще вертикальные линии
TERMINAL_LINE_LENGTH = 350  # длина линии в пикселах, при которой мы однозначно получаем угол здания
DISTANCE_DIFFERENCE_THRESHOLD = 4  # расстояние в метрах, если разница между точками с правой и левой стороны больше этого значения, то мы нашли угол
CENTER_L_BORDER_X = -1
CENTER_R_BORDER_X = -1
REMOVE_DISTANCE_PIXEL_DX = 5
LINE_TO_IMG = True


def removeDistanceBeyondBorder(depth, kp, flag):
    newDepth = []
    newDepth_overseas = []
    if flag == 'r' :
        if CENTER_R_BORDER_X >- 1:
            for j, point in enumerate(kp):
                if point.pt[0] <= CENTER_R_BORDER_X + REMOVE_DISTANCE_PIXEL_DX:
                    newDepth.append(depth[j])
                else:
                    newDepth_overseas.append(depth[j])
        else:
            print('r no border?')
    elif flag =='l':
        if CENTER_L_BORDER_X > - 1:
            for j, point in enumerate(kp):
                if point.pt[0] >= CENTER_L_BORDER_X - REMOVE_DISTANCE_PIXEL_DX:
                    newDepth.append(depth[j])
                else:
                    newDepth_overseas.append(depth[j])
        else:
            print('l no border?')
    return newDepth, newDepth_overseas

# def splitDistanceFromCenter(depth, kp):
#     newDepthRight = []
#     newDepthLeft = []
#     for j, point in enumerate(kp):
#         if point.pt[0] <= CENTER_R_BORDER_X + REMOVE_DISTANCE_PIXEL_DX:
#             newDepthRight.append(depth[j])
#
#     elif flag =='l':
#         if CENTER_L_BORDER_X > - 1:
#             for j, point in enumerate(kp):
#                 if point.pt[0] >= CENTER_L_BORDER_X - REMOVE_DISTANCE_PIXEL_DX:
#                     newDepth.append(depth[j])
#     return newDepth


def detectBorder(kp, depth, img):

    rFlag = False
    lFlag = False
    roofFlag = False
    lines, newImg = detectLine(img)
    i1_2 = newImg
    h, v = splitLines(lines)
    # if LINE_TO_IMG and img is not None:
        # i1_2 = drawLines(newImg, h, horizontal_color)
        # i1_2 = drawLines(i1_2, v, vertical_color)
    if len(depth) > MIN_DEPTH_LEN:
        r, l = detectAllBorders(v, kp, depth)
        roof = detectRoofBorder(h, kp, depth)
        if LINE_TO_IMG and img is not None:
            # i1_2 = drawLines(newImg, r, colorRB)# i1_2
            # i1_2 = drawLines(i1_2, l, colorLB)
            i1_2 = drawLines(newImg, r, vertical_color)# i1_2
            i1_2 = drawLines(i1_2, l, vertical_color)
            i1_2 = drawLines(i1_2, roof, horizontal_color)

        center = img.shape[:2]
        if len(r) != 0:
            rFlag, centerBordersR, meanCenterR = borderAtCenter(r, (center[1] / 2, center[0] / 2))
            global CENTER_R_BORDER_X
            CENTER_R_BORDER_X = meanCenterR
            # i1_2 = drawLines(i1_2, centerBordersR, horizontal_color)
            if(rFlag):
                cv.putText(i1_2, "RIGHT BORDER!!!", (450, 500), cv.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 4, cv.LINE_AA)

        if len(l) != 0:
            lFlag, centerBordersL, meanCenterL = borderAtCenter(l, (center[1] / 2, center[0] / 2))
            global CENTER_L_BORDER_X
            CENTER_L_BORDER_X = meanCenterL
            # i1_2 = drawLines(i1_2, centerBordersL, vertical_color)
            if lFlag:
                cv.putText(i1_2, "LEFT BORDER!!!", (450, 700), cv.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 4, cv.LINE_AA)
        if len(roof)!= 0:
            roofFlag, roofBorder = roofAtCenter(roof, (center[1] / 2, center[0] / 2))
            i1_2 = drawLines(i1_2, roofBorder, colorRoof)
            if roofFlag:
                cv.putText(i1_2, "ROOF !!!", (450, 300), cv.FONT_HERSHEY_PLAIN, 6, (255, 0, 0), 4, cv.LINE_AA)

    # else:
    #     i1_2 = newImg
    resultImg = i1_2
    return rFlag, lFlag, roofFlag, resultImg


def drawLines(image, lines, color):
    img = image.copy()
    if lines is not None:
        for i in lines:
            l = i[0]
            cv.line(img, (l[0], l[1]), (l[2], l[3]), color, 3, cv.LINE_AA)
    return img


def detectLine(img):

    test = cv.GaussianBlur(img, (3, 3), 0)
    test = cv.Canny(test, 50, 200, None, 3)
    # lines = cv.HoughLinesP(testc, 1, np.pi / 180, 50, None, 60, 20)
    lines = cv.HoughLinesP(test, 1, np.pi / 180, 50, None, 100, 25)
    test = cv.cvtColor(test, cv.COLOR_GRAY2BGR)
    return lines, test


def videoWrite(imgArr, size, name=''):
    out = cv.VideoWriter(name + '.avi', cv.VideoWriter_fourcc(*'DIVX'), 3, size)
    for img in imgArr:
        out.write(img)
    out.release()


def splitLines(lines):
    horizontal = []
    vertical = []
    if lines is not None:
        for i in lines:
            l = i[0]
            if np.abs((l[0] - l[2])) >= np.abs((l[1] - l[3])):
                if np.abs((l[1] - l[3]))/np.abs((l[0] - l[2])) < 0.1: #tg
                    horizontal.append(i)
            else:
                if np.abs((l[0] - l[2]))/np.abs((l[1] - l[3])) < 0.55:
                    vertical.append(i)
    return horizontal, vertical



def detectAllBorders(vline, kp, depth):
    leftBorder = []
    rightBorder = []
    for i in vline:
        l = i[0]
        tmp_x = (l[0] + l[2]) / 2
        rightDepthValue = []
        leftDepthValue = []
        for j, point in enumerate(kp):
            if tmp_x < point.pt[0] <= tmp_x + PIXEL_DX:
                rightDepthValue.append(depth[j])
            elif tmp_x > point.pt[0] >= tmp_x - PIXEL_DX:
                leftDepthValue.append(depth[j])

        if len(leftDepthValue) != 0 and len(rightDepthValue) != 0:
            # a = list(filter(lambda x: 3 < x, rightDepthValue))
            # b = list(filter(lambda x: 3 < x, leftDepthValue))
            # if len(a) > 1:
            #     rightDepthValue = a
            # if len(b) > 1:
            #     leftDepthValue = b
            meanRight = np.mean(rightDepthValue)
            meanLeft = np.mean(leftDepthValue)

            if np.abs(meanLeft - meanRight) > DISTANCE_DIFFERENCE_THRESHOLD:
                if meanLeft > meanRight:
                    leftBorder.append(i)
                else:
                    rightBorder.append(i)
    return rightBorder, leftBorder


def removeOutliers(array):
    sigma = np.std(array)
    mean = np.mean(array)
    newArray = list(filter(lambda x: sigma - mean <= x <= mean + sigma, array))
    return newArray


def borderAtCenter(borders, center):
    res = False
    centerBorders = []
    meanCenter = 0
    for border in borders:
        l = border[0]
        tmp_x = (l[0] + l[2]) / 2
        if center[0] - PIXEL_CENTER_DX <= tmp_x <= center[0] + PIXEL_CENTER_DX:
            meanCenter+=tmp_x
            centerBorders.append(border)
            if (len(centerBorders) > 2) or np.abs(l[2] - l[3]) > TERMINAL_LINE_LENGTH:
                res = True
                meanCenter /= len(centerBorders)
                break

    return res, centerBorders, meanCenter


def roofAtCenter(borders, center):
    res = False
    centerBorders = []
    # print('center', center, center[1] - PIXEL_CENTER_DX, center[1] + PIXEL_CENTER_DX, center[0] - PIXEL_DX, center[0] + PIXEL_DX)
    for border in borders:
        l = border[0]
        tmp_y = (l[1] + l[3]) / 2
        tmp_x =  (l[0] + l[2]) / 2
        if center[1] - PIXEL_CENTER_DX <= tmp_y <= center[1] + PIXEL_CENTER_DX and center[0] - PIXEL_DX <= tmp_x <= center[0] + PIXEL_DX:
            centerBorders.append(border)
            # print('append r center', tmp_y, tmp_x)
            if (len(centerBorders) > 3) or np.abs(l[2] - l[3]) > TERMINAL_LINE_LENGTH:
                res = True
                break
    return res, centerBorders

def detectRoofBorder(hline, kp, depth):
    topBorder = []
    for i in hline:
        l = i[0]
        tmp_y = (l[1] + l[3]) / 2
        tmp_x = (l[0] + l[2])/2
        if(l[0] > l[2]):
            print('ALERT 1 > 2')
        topDepthValue = []
        bottomDepthValue = []
        for j, point in enumerate(kp):
            if tmp_y < point.pt[1] <= tmp_y + PIXEL_DX and l[0] <= point.pt[0] <= l[2]:
                bottomDepthValue.append(depth[j])
            elif tmp_y > point.pt[1] >= tmp_y - PIXEL_DX  and l[0] <= point.pt[0] <= l[2]:
                topDepthValue.append(depth[j])
        if len(topDepthValue) != 0 and len(bottomDepthValue) != 0:
            meanTop = np.mean(topDepthValue)
            # print('BOTTOm', bottomDepthValue)

            # print('before mean len list',np.mean(bottomDepthValue), len(bottomDepthValue), bottomDepthValue)
            bottomDepthValue = list(filter(lambda x: MAX_DRON_DISTANCE > x, bottomDepthValue))

            # print('after mean len list', np.mean(bottomDepthValue), len(bottomDepthValue), bottomDepthValue)
            if len(bottomDepthValue)!= 0:
                meanBottom = np.mean(bottomDepthValue)
            else:
                continue
            if np.abs(meanTop - meanBottom) > DISTANCE_DIFFERENCE_THRESHOLD and meanBottom <= MAX_DRON_DISTANCE:
                if meanTop > meanBottom:
                    topBorder.append(i)
                    # print('append top b', meanTop, meanBottom)
    # print('len ', len(topBorder))
    return topBorder
