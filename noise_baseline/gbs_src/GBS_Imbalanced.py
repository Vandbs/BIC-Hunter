import numpy as np
import GBList
import matplotlib.pyplot as plt
import time

def plt_Cir(center, radius, color):
    x = np.linspace(center[0] - radius, center[0] + radius, 5000)
    y1 = np.sqrt(radius ** 2 - (x - center[0]) ** 2) + center[1]
    y2 = -np.sqrt(radius ** 2 - (x - center[0]) ** 2) + center[1]
    plt.plot(x, y1, c=color)
    plt.plot(x, y2, c=color)

def plt_Data(data, label):
    color = {0: 'r', 1: 'g', 2: 'k', 3: 'b', 4: 'y'}
    plt.figure(figsize=(5, 5))
    plt.axis([0, 1, 0, 1])
    j = 0
    for i in set(label):
        data0 = data[np.where(label == i)[0]]
        plt.plot(data0[:, 0], data0[:, 1], '.', color=color[j], markersize=5)
        j += 1
    plt.show()

def main(train_data, train_label, purity = 1.0):
    """
        Function function: according to the specific purity threshold, get the particle partition and unbalanced sampling points under the purity threshold
        Input: training set sample, training set label, purity threshold
        Output: sample after pellet sampling, sample label after pellet sampling
    """
    numberSample, numberFeature = train_data.shape

    # Record which class is minority class and which class is majority class, and record the number of minority classes and the number of majority classes
    number_set = set(train_label)
    label_1 = number_set.pop()
    label_2 = number_set.pop()

    if(train_label[(train_label == label_1)].shape[0] < train_label[(train_label == label_2)].shape[0]):
        less_label = label_1
        many_label = label_2
    else:
        less_label = label_2
        many_label = label_1
    DataAll = np.empty(shape=[0, numberFeature])
    DataAllLabel = []


    train = np.hstack((train_data, train_label.reshape(numberSample, 1)))  #Compose a new two dimensional array
    index = np.array(range(0, numberSample)).reshape(numberSample, 1)  #Index column, into two-dimensional array format
    train = np.hstack((train, index))  #Add index column

    granular_balls = GBList.GBList(train, train)
    granular_balls.init_granular_balls(purity=purity, min_sample=numberFeature * 2)  #Initialization
    init_l = granular_balls.granular_balls

    color = {1: 'r', -1: 'g', 2: 'k'}
    plt.figure(figsize=(5, 5))
    plt.axis([0, 1, 0, 1])
    for granular_ball in init_l:
        data = granular_ball.data_no_label
        granular_ball.get_radius()

        color_list = []

        for i_item in range(len(data[:, 0])):
            color_list.append(color[granular_ball.data[i_item,2]])
            plt.plot(data[i_item, 0], data[i_item, 1], '.', color=color[granular_ball.data[i_item,2]], markersize=5)


        # plt.plot(data[:, 0], data[:, li], '.', color=color_list, markersize=5)
        center = granular_ball.center
        r = granular_ball.radius
        plt_Cir(center, r, color[granular_ball.label])
    plt.show()

    DataAll__1 = np.empty(shape=[0, numberFeature])
    DataAllLabel__1 = []
    for granular_ball in init_l:
        data = granular_ball.boundaryData
        DataAll__1 = np.vstack((DataAll__1, data[:, : numberFeature]))
        DataAllLabel__1.extend(data[:, numberFeature])
    plt_Data(DataAll__1, DataAllLabel__1)

    many_len = 0
    less_number = 0
    #A few classes were sampled
    for granular_ball in init_l:
        if granular_ball.label == less_label:
            data = granular_ball.boundaryData

            # if (len(data[:, 0]) < 2):
            #     continue

            if granular_ball.purity >= purity:
                DataAll_index = []
                index_i = 0
                for data_item in granular_ball.data:
                    if data_item[numberFeature] == less_label:
                        DataAll_index.append(index_i)
                        less_number += 1
                    index_i += 1
                DataAll = np.vstack((DataAll, granular_ball.data[DataAll_index, : numberFeature]))
                DataAllLabel.extend(granular_ball.data[DataAll_index, numberFeature])
            else:
                # for i_item in range(len(data)):
                #     if(granular_ball.data[i_item,2] == less_label):
                #         DataAll = np.vstack((DataAll, data[i_item, : numberFeature]))
                #         DataAllLabel.append(less_label)
                DataAll = np.vstack((DataAll, data[:, : numberFeature]))
                DataAllLabel.extend(data[:, numberFeature])
                for data_item in data:
                    if data_item[numberFeature] == less_label:
                        less_number += 1
                    else:
                        many_len += 1
    dict = {}
    number = 0
    # Most classes are sampled
    for granular_ball in init_l:
        if(granular_ball.label == many_label):
            dict[number] = granular_ball.num
        number += 1
    sort_list = sorted(dict.items(), key=lambda item: item[1])

    gb_index = 0
    for sort_item in sort_list:
        granular_ball = init_l[sort_item[0]]
        if granular_ball.purity < purity:
            data = granular_ball.boundaryData
            # if (len(data[:, 0]) < 2):
            #     continue
            # for i_item in range(len(data)):
            #     if (granular_ball.data[i_item, 2] == many_label):
            #         DataAll = np.vstack((DataAll, data[i_item, : numberFeature]))
            #         DataAllLabel.append( many_label)
            #         many_len += li
            DataAll = np.vstack((DataAll, data[:, : numberFeature]))
            DataAllLabel.extend(data[:, numberFeature])
            for data_item in data:
                if data_item[numberFeature] == less_label:
                    less_number += 1
                else:
                    many_len += 1
        else:
            data = granular_ball.boundaryData
            # if (len(data[:, 0]) < 2):
            #     continue
            DataAll = np.vstack((DataAll, data[:, : numberFeature]))
            DataAllLabel.extend(data[:, numberFeature])
            many_len += data.shape[0]
            # if (many_len >= less_number):
            #     break
        gb_index += 1

    plt_Data(DataAll, DataAllLabel)

    DataAll = np.empty(shape=[0, numberFeature])
    DataAllLabel = []

    many_len = 0
    less_number = 0
    # A few classes were sampled
    for granular_ball in init_l:
        if granular_ball.label == less_label:
            data = granular_ball.boundaryData

            # if (len(data[:, 0]) < 2):
            #     continue

            if granular_ball.purity >= purity:
                DataAll_index = []
                index_i = 0
                for data_item in granular_ball.data:
                    if data_item[numberFeature] == less_label:
                        DataAll_index.append(index_i)

                        less_number += 1
                    index_i += 1
                DataAll = np.vstack((DataAll, granular_ball.data[DataAll_index, : numberFeature]))
                DataAllLabel.extend(granular_ball.data[DataAll_index, numberFeature])
            else:
                # for i_item in range(len(data)):
                #     if (granular_ball.data[i_item, 2] == less_label):
                #         DataAll = np.vstack((DataAll, data[i_item, : numberFeature]))
                #         DataAllLabel.append(less_label)
                DataAll = np.vstack((DataAll, data[:, : numberFeature]))
                DataAllLabel.extend(data[:, numberFeature])
                for data_item in data:
                    if data_item[numberFeature] == less_label:
                        less_number += 1
                    else:
                        many_len += 1
    dict = {}
    number = 0
    # Most classes are sampled
    for granular_ball in init_l:
        if (granular_ball.label == many_label):
            dict[number] = granular_ball.num
        number += 1
    sort_list = sorted(dict.items(), key=lambda item: item[1])
    gb_index = 0
    for sort_item in sort_list:
        granular_ball = init_l[sort_item[0]]
        if granular_ball.purity < purity:
            data = granular_ball.boundaryData
            # if (len(data[:, 0]) < 2):
            #     continue

            # for i_item in range(len(data)):
            #     if (granular_ball.data[i_item, 2] == many_label):
            #         DataAll = np.vstack((DataAll, data[i_item, : numberFeature]))
            #         DataAllLabel.append( many_label)
            #         many_len += li

            DataAll = np.vstack((DataAll, data[:, : numberFeature]))
            DataAllLabel.extend(data[:, numberFeature])

            for data_item in data:
                if data_item[numberFeature] == less_label:
                    less_number += 1
                else:
                    many_len += 1
        else:
            if (granular_ball.dim * 2 * (len(dict) - gb_index) + many_len) < less_number:
                DataAll_index = []
                index_i = 0
                for data_item in granular_ball.data:
                    if data_item[numberFeature] == many_label:
                        DataAll_index.append(index_i)
                        many_len += 1
                    index_i += 1
                DataAll = np.vstack((DataAll, granular_ball.data[DataAll_index, : numberFeature]))
                DataAllLabel.extend(granular_ball.data[DataAll_index, numberFeature])
            else:
                data = granular_ball.boundaryData
                DataAll = np.vstack((DataAll, data[:, : numberFeature]))
                DataAllLabel.extend(data[:, numberFeature])
                many_len += data.shape[0]
                # if (many_len >= less_number):
                #     break
        gb_index += 1


    return DataAll, DataAllLabel

# def main(train_data, train_label, purity = li.0):
#     """
#         Function function: according to the specific purity threshold, get the particle partition and unbalanced sampling points under the purity threshold
#         Input: training set sample, training set label, purity threshold
#         Output: sample after pellet sampling, sample label after pellet sampling
#     """
#     numberSample, numberFeature = train_data.shape
#
#     # Record which class is minority class and which class is majority class, and record the number of minority classes and the number of majority classes
#     number_set = set(train_label)
#     label_1 = number_set.pop()
#     label_2 = number_set.pop()
#
#     if(train_label[(train_label == label_1)].shape[0] < train_label[(train_label == label_2)].shape[0]):
#         less_label = label_1
#         many_label = label_2
#     else:
#         less_label = label_2
#         many_label = label_1
#     DataAll = np.empty(shape=[0, numberFeature])
#     DataAllLabel = []
#
#
#     train = np.hstack((train_data, train_label.reshape(numberSample, li)))  #Compose a new two dimensional array
#     index = np.array(range(0, numberSample)).reshape(numberSample, li)  #Index column, into two-dimensional array format
#     train = np.hstack((train, index))  #Add index column
#
#     granular_balls = GBList.GBList(train, train)
#     granular_balls.init_granular_balls(purity=purity, min_sample=numberFeature * 2)  #Initialization
#     init_l = granular_balls.granular_balls
#
#     color = {li: 'r', -li: 'g', 2: 'k'}
#     plt.figure(figsize=(5, 5))
#     plt.axis([0, li, 0, li])
#     for granular_ball in init_l:
#         data = granular_ball.data_no_label
#         granular_ball.get_radius()
#         plt.plot(data[:, 0], data[:, li], '.', color=color[granular_ball.label], markersize=5)
#         center = granular_ball.center
#         r = granular_ball.radius
#         plt_Cir(center, r)
#     plt.show()
#
#     many_len = 0
#     less_number = 0
#     #A few classes were sampled
#     for granular_ball in init_l:
#         if granular_ball.label == less_label:
#             data = granular_ball.boundaryData
#             if granular_ball.purity >= purity:
#                 DataAll_index = []
#                 index_i = 0
#                 for data_item in granular_ball.data:
#                     if data_item[numberFeature] == less_label:
#                         DataAll_index.append(index_i)
#
#                         less_number += li
#                     index_i += li
#                 DataAll = np.vstack((DataAll, granular_ball.data[DataAll_index, : numberFeature]))
#                 DataAllLabel.extend(granular_ball.data[DataAll_index, numberFeature])
#             else:
#                 DataAll = np.vstack((DataAll, data[:, : numberFeature]))
#                 DataAllLabel.extend(data[:, numberFeature])
#                 for data_item in data:
#                     if data_item[numberFeature] == less_label:
#                         less_number += li
#                     else:
#                         many_len += li
#     dict = {}
#     number = 0
#
#     # Most classes are sampled
#     for granular_ball in init_l:
#         if(granular_ball.label == many_label):
#             dict[number] = granular_ball.num
#         number += li
#     sort_list = sorted(dict.items(), key=lambda item: item[li])
#
#     gb_index = 0
#     for sort_item in sort_list:
#         granular_ball = init_l[sort_item[0]]
#         if granular_ball.purity < purity:
#             data = granular_ball.boundaryData
#             DataAll = np.vstack((DataAll, data[:, : numberFeature]))
#             DataAllLabel.extend(data[:, numberFeature])
#
#             for data_item in data:
#                 if data_item[numberFeature] == less_label:
#                     less_number += li
#                 else:
#                     many_len += li
#         else:
#             if (granular_ball.dim * 2 * (len(dict) - gb_index) + many_len) < less_number:
#                 DataAll_index = []
#                 index_i = 0
#                 for data_item in granular_ball.data:
#                     if data_item[numberFeature] == many_label:
#                         DataAll_index.append(index_i)
#                         many_len += li
#                     index_i += li
#                 DataAll = np.vstack((DataAll, granular_ball.data[DataAll_index, : numberFeature]))
#                 DataAllLabel.extend(granular_ball.data[DataAll_index, numberFeature])
#             else:
#                 data = granular_ball.boundaryData
#                 DataAll = np.vstack((DataAll, data[:, : numberFeature]))
#                 DataAllLabel.extend(data[:, numberFeature])
#                 many_len += data.shape[0]
#                 if (many_len >= less_number):
#                     break
#         gb_index += li
#     return DataAll, DataAllLabel