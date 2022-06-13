import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

DIR_NAME = 'dataset_04/distance_data/'
FILE_NAME = 'distance_data.txt'



def videoWrite(imgArr, size, dir):
    out = cv.VideoWriter(dir+'12dist.avi', cv.VideoWriter_fourcc(*'DIVX'), 3, size)
    for img in imgArr:
        out.write(img)
    out.release()

def writeDistanseData(depth, filename):
    print(len(depth))
    f = open(filename, 'w')
    for j in depth:
        for i in j:
            f.write(str(i)+' ')
        f.write('\n')
    f.close()
    return

def GraphDistance(filename = ''):
    if(filename == ''):
        f = open(DIR_NAME+FILE_NAME, 'r')
    else:
        f = open(filename, 'r')
        dn = filename.split('/')
        dn = '/'.join(dn[:-1])
        print(dn)
    x = []
    y = []
    y2 = []
    x2 = []
    tmp2 = []
    tmp = []
    for i, line in enumerate(f):
        tmp_ = line.split(' ')
        tmp_ = [float(t) for t in tmp_[:-1]]

        tmp2.clear()
        tmp.clear()
        for j in tmp_:

            if( 6 < j < 18):
                tmp2.append(j)
            if ( j < 200):
                tmp.append(j)
        x.extend([i for a in range(len(tmp))])
        y.extend(tmp)
        x2.extend([i for a in range(len(tmp2))])
        y2.extend(tmp2)
        # if(i > 59):
        #     break;
    f.close()
    px = 1 / plt.rcParams['figure.dpi']

    plt.subplots(figsize=(1600*px, 600*px))
    plt.grid()

    plt.scatter(x,y, s=10, facecolors='none', edgecolors='g')
    plt.xlabel('iteration')
    plt.ylabel('distance, m')
    if (filename == ''):
        plt.savefig(DIR_NAME+'distance_plot.png')
    else:
        plt.savefig(dn + '/_distance_plot<200_a.png')
    plt.clf()

    plt.subplots(figsize=(1600 * px, 600 * px))
    plt.grid()

    plt.scatter(x2, y2, s=10, facecolors='none', edgecolors='g')
    plt.xlabel('iteration')
    plt.ylabel('distance, m')

    # plt.show()
    if (filename == ''):
        plt.savefig(DIR_NAME+'distance_plot20_060_34.png')
    else:
        plt.savefig(dn + '/_distance_plot<20_a.png')
    plt.clf()
    plt.close()

def rotationGraph(rotation, dir_path):
    pitch = []
    yaw = []
    roll = []
    for i in rotation:
        pitch.append(i[0])
        yaw.append(i[1])
        roll.append(i[2])
    x = [i for i in range(0, len(pitch))]
    # print(pitch, yaw, roll, x, sep='\n')
    px = 1 / plt.rcParams['figure.dpi']
    plt.subplots(figsize=(1600 * px, 600 * px))
    plt.ylabel('degrees')
    plt.xlabel('d, m')
    plt.plot(x, pitch, 'b', label='pitch ' )#y
    plt.plot(x, yaw, 'y', label='yaw ') #z
    plt.plot(x, roll, 'r', label='roll ') #x
    plt.legend()
    plt.savefig(dir_path + '/graphPYR.png')



    ####

def GraphDistanceTrue(trueDistance, filename = ''):
    if(filename == ''):
        f = open(DIR_NAME+FILE_NAME, 'r')
    else:
        f = open(filename, 'r')
        dn = filename.split('/')
        dn = '/'.join(dn[:-1])
        print(dn)
    y2 = []
    x2 = []
    tmp2 = []
    n = 18
    for i, line in enumerate(f):
        tmp_ = line.split(' ')
        tmp_ = [float(t) for t in tmp_[:-1]]
        tmp2.clear()

        for j in tmp_:
            if( 0 < j <= n):
                tmp2.append(j)
            elif(j>n):
                tmp2.append(n+1)
            else:
                tmp2.append(-1)
        x2.extend([i for a in range(len(tmp2))])
        y2.extend(tmp2)
        print(i, len(tmp_), len(tmp2), len(x2), len(y2), sep=' ')

    f.close()
    px = 1 / plt.rcParams['figure.dpi']

    plt.subplots(figsize=(1600 * px, 600 * px))
    plt.grid()

    plt.scatter(x2, y2, s=10, facecolors='none', edgecolors='g')
    plt.xlabel('iteration')
    plt.ylabel('distance, m')

    if (filename == ''):
        plt.savefig(DIR_NAME+'distance_plot_noFile.png')
    else:
        plt.savefig(dn + '/_distance_plot.png')
    plt.clf()
    plt.close()

##TODO: TRUE DIST?
def graphDistance(true_distance, distanceMean, filter_mean, newMean, dataset_dir=''):
    true_distance = true_distance / 100

    px = 1 / plt.rcParams['figure.dpi']
    plt.subplots(figsize=(1600 * px, 600 * px))
    plt.grid()
    y2 = list(true_distance[:len(distanceMean)])
    print(len(y2))
    maxD = 40
    for i, a in enumerate(distanceMean):
        if a > maxD:
            distanceMean[i] = maxD
        elif a < 0:
            distanceMean[i] = -1
        if (i < len(y2)):
            if (y2[i] > maxD):
                y2[i] = maxD
        else:
            print(i)
            y2.extend([-1])
    # distanceMean = list(filter(lambda x: x < 20, distanceMean))
    x = [a for a in range(len(distanceMean))]
    # dist = np.linalg.norm(distanceMean - filter_mean)
    # print(dist)
    plt.plot(x, distanceMean,'--b', label='exp', alpha=0.7)
    plt.plot(x, filter_mean,'-',color='orange', label='filter',  alpha=0.7)
    plt.plot(x, y2, '-.g',  label='true', alpha=0.7)  # Plot more data on the axes...
    plt.xlabel('iteration')
    plt.ylabel('distance, m')
    plt.legend(prop={"size":16})
    plt.savefig(dataset_dir + '/distance_data/markers_4.png')