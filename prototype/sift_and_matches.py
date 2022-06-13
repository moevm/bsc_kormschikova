import cv2 as cv

THRESHOLD_OF_MATCHES = 0.25 #normal = 0.25

def sift_algorithm(img, save=False, save_filepath='s.png'):
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    if save:
        img = cv.drawKeypoints(img, kp, img)
        cv.imwrite(save_filepath, img)
    return kp, des


def detect_matches(firstDes, secondDes):
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(firstDes, secondDes, k=2)

    goodMatches = []
    # badm = []
    # TODO: for skip N match del
    # j = 240
    for i, (m, n) in enumerate(matches):
        if m.distance < THRESHOLD_OF_MATCHES * n.distance:
            # if j != 0:
            #     j -= 1
            #     continue
            goodMatches.append(matches[i])
            #TODO: for only N match del
            # if len(goodMatches) > 10:
            #     return goodMatches
        # else:
        #     badm.append(matches[i])
    return goodMatches#, matches, badm


def save_matches_img(matches, img1, kp1, img2, kp2, filepath='matches.jpg'):
    matches = matches[:int(len(matches)/4)]
    matchesMask = [[1, 0] for i in range(len(matches))]
    draw_params = dict(#matchColor=(0, 255, 0),
                       #singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask[:],
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img_3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches[:], None, **draw_params)
    cv.imwrite(filepath, img_3)
    # cv.imshow('a', img_3)
    # cv.waitKey(0)
    return


def draw_matches_img(matches, img1, kp1, img2, kp2):
    matchesMask = [[1, 0] for i in range(len(matches))]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask[:],
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img_3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches[:], None, **draw_params)
    return img_3