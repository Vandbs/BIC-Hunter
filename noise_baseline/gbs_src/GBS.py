import numpy as np
from replication.noise_baseline.gbs_src import GBList

def main(train_data, train_label, purity = 1.0):
    """
        Function function: according to the specific purity threshold, obtain the particle partition and sampling point under the purity threshold
        Input: training set sample, training set label, purity threshold
        Output: sample after pellet sampling, sample label after pellet sampling
    """
    numberSample, numberFeature = train_data.shape
    train = np.hstack((train_data, train_label.reshape(numberSample, 1)))  # Compose a new two dimensional array
    index = np.array(range(0, numberSample)).reshape(numberSample, 1)  # Index column, into two-dimensional array format
    train = np.hstack((train, index))  # Add index column

    granular_balls = GBList.GBList(train, train)
    granular_balls.init_granular_balls(purity=purity, min_sample=numberFeature * 2)  # initialization
    init_l = granular_balls.granular_balls

    DataAll = np.empty(shape=[0, numberFeature])
    DataAllLabel = []
    for granular_ball in init_l:
        data = granular_ball.boundaryData
        DataAll = np.vstack((DataAll, data[:, : numberFeature]))
        DataAllLabel.extend(data[:, numberFeature])
    return DataAll, DataAllLabel