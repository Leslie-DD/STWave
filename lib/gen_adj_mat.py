from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    # 初始化距离矩阵:
    # 创建一个大小为 (num_sensors, num_sensors) 的矩阵 dist_mx，初始值为无穷大（inf），表示传感器之间的距离。
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf

    # Builds sensor id to index map.
    # 构建传感器 ID 到索引的映射:
    # 使用字典 sensor_id_to_ind 将每个传感器 ID 映射到矩阵的行/列索引
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    # 填充距离矩阵:
    # 遍历 distance_df 中的每一行，根据 from 和 to 列的传感器 ID，将对应的距离填入 dist_mx。
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    # 计算邻接矩阵:
    # 计算所有有限距离的标准差（std），并使用高斯核函数（exp(-distance^2 / std^2)）将距离转换为相似度，生成邻接矩阵 adj_mx。
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    # 稀疏化邻接矩阵:
    # 将 adj_mx 中小于 normalized_k 的值置为 0，以减少矩阵的稠密度。
    adj_mx[adj_mx < normalized_k] = 0

    print(adj_mx.shape)
    return sensor_ids, sensor_id_to_ind, adj_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_filename', type=str, default='../data/PeMSD8/graph_sensor_ids.txt',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--distances_filename', type=str, default='../data/PeMSD8/PEMS08.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_npy_filename', type=str, default='../data/PeMSD8/adj_mat.npy',
                        help='Path of the output file.')
    args = parser.parse_args()

    with open(args.sensor_ids_filename) as f:
        sensor_ids = f.read().strip().split(',')
    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})
    normalized_k = args.normalized_k
    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k)
    # Save to pickle file.
    # with open(args.output_pkl_filename, 'wb') as f:
        # pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
        # pickle.dump(adj_mx, f, protocol=2)


    # save to npy file
    np.save(args.output_npy_filename, adj_mx)
