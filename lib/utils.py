import pywt
import torch
import numpy as np

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        wape = np.divide(np.sum(mae), np.sum(label))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def _compute_loss(y_true, y_predicted):
        return masked_mae(y_predicted, y_true, 0.0)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def seq2instance(data, P, Q):
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, nodes, dims))
    y = np.zeros(shape = (num_sample, Q, nodes, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def disentangle(x, w, j):
    x = x.transpose(0,3,2,1) # [S,D,N,T]
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0]]
    for i in range(len(coef)-1):
        coefl.append(None)
    coefh = [None]
    for i in range(len(coef)-1):
        coefh.append(coef[i+1])
    xl = pywt.waverec(coefl, w).transpose(0,3,2,1)
    xh = pywt.waverec(coefh, w).transpose(0,3,2,1)

    return xl, xh

def load_data(filepath, P, Q, train_ratio, test_ratio, log):
    # Traffic
    original_data = np.load(filepath)['data']
    log_string(log, f'Shape of original:    {original_data.shape}')
    Traffic = np.load(filepath)['data'][...,:1]
    log_string(log, f'Shape of data:        {Traffic.shape}')
    num_step = Traffic.shape[0]
    log_string(log, f'  num steps of data:  {num_step}\n')

    # temporal embedding 生成时间序列数据的时间嵌入
    TE = np.zeros([num_step, 2]) # 创建一个形状为 [num_step, 2] (num_step 行，2 列)的二维数组，所有元素初始化为 0。

    # 一天有 288 个时间间隔（每 5 分钟一个间隔：24 小时 × 60 分钟 ÷ 5 = 288）
    # i % 288 计算每个时间步在一天中的时间间隔索引
    TE[:,1] = np.array([i % 288 for i in range(num_step)]) # 填充 TE 的第二列（TE[:,1]），表示一天中的时间步。

    # i // 288 计算当前时间步属于第几天（因为一天有 288 个时间间隔）。
    # (i // 288) % 7 将天数映射到 0 到 6 的范围，表示一周中的星期几。
    TE[:,0] = np.array([(i // 288) % 7 for i in range(num_step)]) # 填充 TE 的第一列（TE[:,0]），表示星期几。

    # 这里的 TE 是一个二维数组，第一列表示星期几（0 到 6），第二列表示一天中的时间步（0 到 287）。
    log_string(log, f'Shape of TE:          {TE.shape}')


    # 将 TE 数组扩展以匹配 Traffic 数据的空间维度
    # np.expand_dims(TE, 1) 在 TE 的第二维添加一个新轴，使其形状变为 [num_step, 1, 2]。
    # np.repeat(..., Traffic.shape[1], 1) 沿着第二维（空间节点数）复制时间嵌入，生成形状为 [num_step, Traffic.shape[1], 2] 的 TE_tile。
    TE_tile = np.repeat(np.expand_dims(TE, 1), Traffic.shape[1], 1)

    log_string(log, f'Shape of TE_tile:     {TE_tile.shape}\n')

    # train/val/test 
    train_steps = round(train_ratio * num_step)
    test_steps = round(test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    trainData, trainTE = Traffic[: train_steps], TE_tile[: train_steps]
    valData, valTE = Traffic[train_steps : train_steps + val_steps], TE_tile[train_steps : train_steps + val_steps]
    testData, testTE = Traffic[-test_steps :], TE_tile[-test_steps :]


    # X, Y
    trainX, trainY = seq2instance(trainData, P, Q)
    valX, valY = seq2instance(valData, P, Q)
    testX, testY = seq2instance(testData, P, Q)

    trainXTE, trainYTE = seq2instance(trainTE, P, Q)
    valXTE, valYTE = seq2instance(valTE, P, Q)
    testXTE, testYTE = seq2instance(testTE, P, Q)

    # derive temporal embedding
    trainTE = np.concatenate([trainXTE, trainYTE], axis=1)
    valTE = np.concatenate([valXTE, valYTE], axis=1)
    testTE = np.concatenate([testXTE, testYTE], axis=1)

    # normalization
    mean, std = np.mean(trainX), np.std(trainX)

    log_string(log, f'Shape of Train Data:      {trainX.shape}')
    log_string(log, f'Shape of Validation Data: {valX.shape}')
    log_string(log, f'Shape of Test Data:       {testX.shape}\n')

    log_string(log, f'Mean: {mean} & Std: {std}\n')
    
    return trainX, trainY, trainTE, valX, valY, valTE, testX, testY, testTE, mean, std, trainData[...,0]