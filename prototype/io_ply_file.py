import numpy as np
from pathlib import Path

def read_ply_file(filename):
    header_lines = 0

    file_path = Path(filename)
    file = open(str(file_path.absolute()), "r")

    for line in file:
        header_lines += 1
        if line == "end_header\n":
            break

    points = np.genfromtxt(file_path.absolute(), skip_header=header_lines)

    return points

def write_ply_file(filename, points, offset):
    file_path = Path(filename)
    file = open(str(file_path.absolute()), "w")
    file_header = ["ply\n", "format ascii 1.0\n", "element vertex " + str(points.shape[0]) + "\n",
    "property float x\n", "property float y\n", "property float z\n", "property uchar red\n", "property uchar green\n" , "property uchar blue\n"
                   "end_header\n"]
    # file_header = ["ply\n", "format ascii 1.0\n", "element vertex " + str(points.shape[0]) + "\n",
    #                "property float x\n", "property float y\n", "property float z\n",
    #                                          "end_header\n"]
    file.writelines(file_header)
    for point in points:
        file.write(str(point[0]+offset[0]) + " " + str(point[1]+offset[1]) + " " + str(point[2]+offset[2]) + " " + str(int(point[3]/100 % 256))+ " " + str(int(point[4]/100 % 256))+ " " + str(int(point[5]/100% 256))+ "\n")
        # file.write(str(point[0]+offset[0]) + " " + str(point[1]+offset[1]) + " " + str(point[2]+offset[2]) + " " + str(int(point[3]))+ " " + str(int(point[4]))+ " " + str(int(point[5]))+ "\n")

    file.close()

def create_point_cloud(vertecies_filename, point_cloud_filename, offset = [0,0,0]):
    house_vertecies = np.zeros((1, 6))

    file = open(vertecies_filename, "r")
    lines = [line[:-2] for line in file]
    file.close()
    for i in range(len(lines)):
        frame_vertecies = np.array([float(n) * 100.0 for n in lines[i].split(" ") if n and n != "\n"]) #for normal view in meshlab
        # frame_vertecies = np.array([float(n) for n in lines[i].split(" ") if n and n != "\n"])
        frame_vertecies = np.reshape(frame_vertecies, newshape=(-1, 6))
        for frame_vertex in frame_vertecies:
            house_vertecies = np.append(house_vertecies, np.reshape(frame_vertex, (1, 6)), axis=0)
    house_vertecies = np.where(abs(house_vertecies) <= 30000, house_vertecies, 30000)
    write_ply_file(point_cloud_filename, house_vertecies[1:], offset)####1??? RETURN LATER
