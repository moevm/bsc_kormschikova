import numpy as np
import io_ply_file
import time


def hausdorff_dist(points_a, points_b):
    maxDist = -1
    start_time = time.time()

    for point_a in points_a:
        minDist = -1
        for point_b in points_b:
            tmp = point_a-point_b
            tmp = np.linalg.norm(tmp)
            if minDist == -1:
                minDist = tmp
            if tmp < minDist:
                minDist = tmp
        if maxDist < minDist:
            maxDist = minDist

    print("--- %s seconds ---" % (time.time() - start_time))
    return maxDist


point_dist = io_ply_file.read_ply_file("cloudPoint/FakePointCloud/fake_house.ply")
point_true = io_ply_file.read_ply_file("cloudPoint/FakePointCloud/test_house.ply")

# print(p1, p2)
hausdorff_dist(point_dist, point_true)