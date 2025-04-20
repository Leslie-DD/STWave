import pickle

import numpy as np

"""
下面几个方法主要用来检视 npz, npy, pkl 文件的结构和内容
"""
def inspect_npz_file(npz_file_path = "../data/PeMSD4/PEMS04.npz"):
    # 加载 NPZ 文件
    data = np.load(npz_file_path)

    # 查看包含的数组名称（键名）
    print(data.files)  # 输出: ['arr1', 'arr2', ...]

    # 查看某个数组的内容
    print(data["data"])  # 输出对应的数组

def inspect_npy_file(npy_file_path = "../data/PeMSD4/adj.npy"):
    # 加载 NPY 文件
    data = np.load(npy_file_path)

    # 查看数据的形状和类型
    print(data.shape)  # 输出: (num_samples, num_features)
    print(type(data))  # 输出: <class 'numpy.ndarray'>

    # 查看前几行数据
    print(data[:5])  # 输出前5行数据
    # np.set_printoptions(threshold=np.inf)
    # print(data)      # 输出所有数据

def inspect_pkl_file(pkl_file_path = "../data/PeMSD3/adj_mat.pkl"):
    # 加载 PKL 文件
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
        # 查看数据的形状和类型
        print(data.shape)  # 输出: (num_samples, num_features)
        print(type(data))  # 输出: <class 'numpy.ndarray'>

        # 查看前几行数据
        print(data[:5])  # 输出前5行数据
    # # 如果是字典，查看所有键
    # if isinstance(data, dict):
    #     print("字典键:", data.keys())
    #
    # # 如果是对象，查看属性
    # elif hasattr(data, "__dict__"):
    #     print("对象属性:", vars(data).keys())
    #
    # # 如果是 NumPy 数组或 Pandas DataFrame
    # elif hasattr(data, "shape"):  # NumPy 数组
    #     print("数组形状:", data.shape)
    # elif hasattr(data, "columns"):  # Pandas DataFrame
    #     print("DataFrame 列名:", data.columns)


inspect_npy_file()