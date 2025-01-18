import numpy as np
import GBList
import time

def main(train_data, train_label, min_purity = 0.51, max_purity=1.0):
    """
        Function function: according to a certain range of purity threshold, get the particle partition under each purity threshold
        Input: training set sample, training set label, minimum purity threshold, maximum purity threshold
        Output: sample after sampling within each purity threshold range, sample label after sampling within each purity threshold range
    """
    numberSample, numberFeature = train_data.shape
    train = np.hstack((train_data, train_label.reshape(numberSample, 1)))  # Compose a new two dimensional array
    index = np.array(range(0, numberSample)).reshape(numberSample, 1)  # Index column, into two-dimensional array format
    train = np.hstack((train, index))  # Add index column

    granular_balls = GBList.GBList(train, train)
    granular_balls.init_granular_balls_dict(min_purity=min_purity, max_purity=max_purity, min_sample=numberFeature * 2)  # initialization
    init_l_dict = granular_balls.dict_granular_balls

    dict_Data = {}
    dict_Data_Label = {}
    item = min_purity
    for init_l in init_l_dict.values():
        print(init_l)
        DataAll = np.empty(shape=[0, numberFeature])
        DataAllLabel = []
        for granular_ball in init_l:
            data = granular_ball.boundaryData
            DataAll = np.vstack((DataAll, data[:, : numberFeature]))
            DataAllLabel.extend(data[:, numberFeature])
        dict_Data[item] = DataAll
        dict_Data_Label[item] = DataAllLabel
        item += 0.01
    return dict_Data, dict_Data_Label