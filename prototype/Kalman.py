
import numpy as np
from numpy import matmul as mm
from copy import copy

START_VALUE = 10
RESET_VALUE = 10
P_i_1 = np.array([[0.1, 0],
                  [0, 0.1]])
X_i_1 = np.array([[START_VALUE],
                  [0]])
K_ALL = []


def kf(X=np.array([[]]),F=np.array([[]]), P=np.array([[]]), Q=np.array([[]]),
       Gamma=np.array([[]]), Z=np.array([[]]), H=np.array([[]]), R=np.array([[]])):
    X_prior = mm(F, X)
    P_prior = mm(mm(F, P), np.transpose(F)) + mm(mm(Gamma, Q), np.transpose(Gamma))

    y = Z - mm(H, X_prior)

    S = mm(mm(H, P_prior), np.transpose(H)) + R
    K = mm(mm(P_prior, np.transpose(H)), np.linalg.inv(S))  # коррекция предсказания по измерениям

    X_posterior = X_prior + mm(K, y)
    P_posterior = mm((np.eye(len(X)) - mm(K, H)), P_prior)

    return X_posterior, P_posterior, K


def reset_kalman(start_value=START_VALUE):
    global P_i_1, X_i_1, START_VALUE
    P_i_1 = np.array([[0.1, 0],
                      [0, 0.1]])
    X_i_1 = np.array([[start_value],
                      [0]])
    START_VALUE = start_value
    return


def step_kalman(delta_t, sensor_data, prev_sensor_data):
    global P_i_1, X_i_1, K_ALL
    H = np.array([[1, 0],
                  [0, 1]])
    R = np.eye(2)
    Q = np.array([[1, 0],
                  [0, 1]])
    G = np.array([[]])
    U = np.array([[]])
    Z = np.array([[sensor_data],
                  [sensor_data - prev_sensor_data]])
    Gamma = np.array([[delta_t, 0],
                      [0, delta_t]])
    F = np.array([[1, delta_t],
                  [0, 1]])
    X_i, P_i, K_i = kf(X_i_1, F, P_i_1, Q, Gamma, Z, H, R)
    r = X_i[0][0]
    X_i_1 = copy(X_i)
    P_i_1 = copy(P_i)
    # K_ALL.append(np.max(K_i))
    return r