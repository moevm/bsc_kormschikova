import numpy as np


def dataset_unpack(data_path):
    def rotation_and_translations_from_str(str_r_t):
        t = np.zeros(3)
        R = np.zeros(3)

        t[0] = float(str_r_t[0][2:])
        t[1] = float(str_r_t[1][2:])
        t[2] = float(str_r_t[2][2:])

        R[0] = float(str_r_t[3][2:])
        R[1] = float(str_r_t[4][2:])
        R[2] = float(str_r_t[5][2:])

        return (t, R)

    time_steps = np.genfromtxt(data_path + "/" + "times.txt")
    distances = np.genfromtxt(data_path + "/" + "distances.txt")

    ground_truth_t = np.zeros(shape=(len(time_steps), 3))
    ground_truth_R = np.zeros(shape=(len(time_steps), 3))
    ground_truth_vel = np.zeros(shape=(len(time_steps), 3))

    trajectory = np.genfromtxt(data_path + "/" + "trajectory.txt", dtype=str)
    for i in range(len(trajectory)):
        t, R = rotation_and_translations_from_str(trajectory[i])
        ground_truth_t[i] = t.copy()
        ground_truth_R[i] = R.copy() # Pitch, Yaw, Rol

    velocity = np.genfromtxt(data_path + "/" + "velocities.txt", dtype=str)
    for i in range(len(velocity)):
        ground_truth_vel[i][0] = float(velocity[i][0][2:])
        ground_truth_vel[i][1] = float(velocity[i][1][2:])
        ground_truth_vel[i][2] = float(velocity[i][2][2:])

    return (time_steps, distances, ground_truth_t, ground_truth_R, ground_truth_vel)


# import numpy as np
# import cv2 as cv
#
# def dataset_unpack(data_path):
#     def rotation_and_translations_from_str(str_r_t):
#         t = np.zeros(3)
#         R = np.zeros(3)
#
#         t[0] = float(str_r_t[0][2:])
#         t[1] = float(str_r_t[1][2:])
#         t[2] = float(str_r_t[2][2:])
#
#         R[0] = float(str_r_t[3][2:])
#         R[1] = float(str_r_t[4][2:])
#         R[2] = float(str_r_t[5][2:])
#
#         return (t, R)
#     time_steps = np.genfromtxt(data_path + "/" + "times.txt")
#
#     ground_truth_t = np.zeros(shape=(len(time_steps), 3))
#     ground_truth_R = np.zeros(shape=(len(time_steps), 3))
#
#     trajectory = np.genfromtxt(data_path + "/" + "trajectory.txt", dtype = str)
#     for i in range(len(trajectory)):
#         t, R = rotation_and_translations_from_str(trajectory[i])
#         ground_truth_t[i] = t.copy()
#         ground_truth_R[i] = R.copy()
#
#
#     return (time_steps, ground_truth_t, ground_truth_R)