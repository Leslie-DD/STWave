import os

import numpy as np
import matplotlib.pyplot as plt

from metrics import MAE, MAPE, RMSE

plt.style.use('fivethirtyeight')


def get_flow(filename):
    flow_data = np.load(filename)
    return flow_data['data']


def show_pred(test_loader, all_y_true, all_predict_values):
    node_id = 166
    plt.title("no. {} node the first day".format(node_id))
    plt.xlabel("time/5min")
    plt.ylabel("flow")
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_y_true)[:24 * 12, node_id, 0, 0], label='true')
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_predict_values[0])[:24 * 12, node_id, 0, 0], label='ChebNet pred')
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_predict_values[1])[:24 * 12, node_id, 0, 0], label='GCN pred')
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_predict_values[2])[:24 * 12, node_id, 0, 0], label='GAT pred')
    plt.legend()
    plt.savefig("../assets/the first day pred flow in node {}.png".format(str(node_id)), dpi=400)
    plt.show()

    plt.title("no. {} node the two weeks".format(node_id))
    plt.xlabel("time/5min")
    plt.ylabel("flow")
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_y_true)[:, node_id, 0, 0], label='true')
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_predict_values[0])[:, node_id, 0, 0], label='ChebNet pred')
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_predict_values[1])[:, node_id, 0, 0], label='GCN pred')
    plt.plot(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                              all_predict_values[2])[:, node_id, 0, 0], label='GAT pred')
    plt.legend()
    plt.savefig("../assets/the first two weeks pred flow in node {}.png".format(str(node_id)), dpi=400)
    plt.show()

    mae = MAE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               all_y_true),
              test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               all_predict_values[0]))
    rmse = RMSE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_y_true),
                test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_predict_values[0]))
    mape = MAPE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_y_true),
                test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_predict_values[0]))
    print("ChebNet基于原始值的精度指标  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse, mape))

    mae = MAE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               all_y_true),
              test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               all_predict_values[1]))
    rmse = RMSE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_y_true),
                test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_predict_values[1]))
    mape = MAPE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_y_true),
                test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_predict_values[1]))
    print("GCN基于原始值的精度指标  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse, mape))

    mae = MAE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               all_y_true),
              test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                               all_predict_values[2]))
    rmse = RMSE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_y_true),
                test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_predict_values[2]))
    mape = MAPE(test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_y_true),
                test_loader.dataset.recover_data(test_loader.dataset.flow_norm[0], test_loader.dataset.flow_norm[1],
                                                 all_predict_values[2]))
    print("GAT基于原始值的精度指标  mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}".format(mae, rmse, mape))


if __name__ == '__main__':
    pems_dics = {
        "PeMS04": "/home/leslie/Coding/Dissertation/STWave/data/PeMSD4/PEMSD4.npz",
        "PeMS07": "/home/leslie/Coding/Dissertation/STWave/data/PeMSD7/PEMS07.npz",
        "PeMS08": "/home/leslie/Coding/Dissertation/STWave/data/PeMSD8/PEMS08.npz",
    }
    for ds_name, path in pems_dics.items():
        traffic_data = get_flow(path)
        print(f"{ds_name} data size {traffic_data.shape}")
        # 采样某个结点（也就是一个探测器）一天的数据，横坐标是时间（探测器每隔五分钟采一次数据，因此一天中采集了288次数据，可以看到上面的图在横坐标也就是时间上展开大致就是0-288）
        node_id = 120
        plt.plot(traffic_data[:24 * 12, node_id, 0], label="flow")
        if (traffic_data.shape[2] > 1):
            plt.plot(traffic_data[:24 * 12, node_id, 1], label="speed")

        if (traffic_data.shape[2] > 2):
            plt.plot(traffic_data[:24 * 12, node_id, 2], label="other")
        plt.legend(loc=0)

        directory = "../assets"
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(os.path.join(directory, "vis_{}.png".format(ds_name)), dpi=300)
        plt.show()
    # traffic_data = get_flow('/home/leslie/Coding/Dissertation/STWave/data/PeMSD8/PEMS08.npz')
    # print("data size {}".format(traffic_data.shape))
    # # 采样某个结点（也就是一个探测器）一天的数据，横坐标是时间（探测器每隔五分钟采一次数据，因此一天中采集了288次数据，可以看到上面的图在横坐标也就是时间上展开大致就是0-288）
    # node_id = 120
    # plt.plot(traffic_data[: 24 * 12, node_id, 0], label="flow")
    # plt.plot(traffic_data[: 24 * 12, node_id, 1], label="speed")
    # plt.plot(traffic_data[: 24 * 12, node_id, 2], label="other")
    # plt.legend(loc=0)
    #
    # directory = "../assets"
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    #
    # plt.savefig(os.path.join(directory, "vis.png"))
    # plt.show()
