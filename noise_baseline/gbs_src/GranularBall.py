from collections import Counter
from sklearn.cluster._kmeans import k_means
import numpy as np
import random

class GranularBall:
    def __init__(self, data):  # Data is labeled data, the penultimate column is label, and the last column is index
        self.data = data
        self.data_no_label = data[:, :-2]
        self.num, self.dim = self.data_no_label.shape  # Number of rows, number of columns
        self.center = self.data_no_label.mean(0)  # According to the calculation of row direction, the mean value of all the numbers in each column (that is, the center of the pellet) is obtained
        self.label, self.purity = self.__get_label_and_purity()  # The type and purity of the label to put back the pellet
        self.init_center = self.random_center()  # Get a random point in each tag
        self.label_num = len(set(data[:, -2]))
        self.boundaryData = None
        self.radius = None

    def random_center(self):
        """
            Function function: saving centroid
            Return: centroid of all generated clusters
        """
        center_array = np.empty(shape=[0, len(self.data_no_label[0, :])])
        for i in set(self.data[:, -2]):
            data_set = self.data_no_label[self.data[:, -2] == i, :]  # A label is equal to the set of all the points to label a point
            random_data = data_set[random.randrange(len(data_set)), :]  # A random point in the dataset
            center_array = np.append(center_array, [random_data], axis=0)  # Add to the line
        return center_array

    def __get_label_and_purity(self):
        """
           Function function: calculate purity and label type
       """
        count = Counter(self.data[:, -2])  # Counter, put back the number of class tags
        label = max(count, key=count.get)  # Get the maximum number of tags
        purity = count[label] / self.num  # Purity obtained, percentage of tags
        return label, purity

    def get_radius(self):
        """
           Function function: calculate radius
       """
        diffMat = np.tile(self.center, (self.num, 1)) - self.data_no_label
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        self.radius = distances.sum(axis=0) / self.num

    def split_clustering(self):
        """
           Function function: continue to divide the granule into several new granules
           Output: new pellet list
       """
        Clusterings = []
        ClusterLists = k_means(X=self.data_no_label, init=self.init_center, n_clusters=self.label_num)
        data_label = ClusterLists[1]  # Get a list of tags
        for i in range(self.label_num):
            Cluster_data = self.data[data_label == i, :]
            if len(Cluster_data) > 1:
                Cluster = GranularBall(Cluster_data)
                Clusterings.append(Cluster)
        return Clusterings



    # def split_clustering(self, purity_y):
    #     """
    #        Function function: continue to divide the granule into several new granules
    #        Output: new pellet list
    #    """
    #
    #     # 判断粒球的标签和纯度
    #     def get_label_and_purity(gb):
    #         # 矩阵的行数
    #         num = gb.shape[0]
    #         # print(data.shape)
    #
    #         # 分离不同标签数据
    #         len_label = np.unique(gb[:, -2], axis=0)
    #         # print(len_label)
    #         gb_label_temp = {}
    #         for label in len_label.tolist():
    #             # 分离不同标签距离
    #             gb_label_temp[sum(gb[:, -2] == label)] = label
    #         # print(gb_label_temp)
    #         # 粒球中最多的一类数据占整个的比例
    #         purity = max(gb_label_temp.keys()) / num if num else li.0
    #         label = gb_label_temp[max(gb_label_temp.keys())]
    #         # print(label)
    #         # print(purity)
    #         # 标签、纯度
    #         return label, purity
    #
    #     def calculate_distances(data, p):
    #         return ((data - p) ** 2).sum(axis=0) ** 0.5
    #
    #     def splits(purity, gb_dict):
    #         gb_len = li
    #         # print(purity)
    #         while True:
    #             ball_number_1 = len(gb_dict)
    #             gb_dict_single = gb_dict.copy()  # 复制一个临时list，接下来再遍历取值
    #             for i in range(0, gb_len):
    #                 gb_single = {}
    #                 # 取字典数据，包括键值
    #                 gb_dict_temp = gb_dict_single.popitem()
    #                 gb_single[gb_dict_temp[0]] = gb_dict_temp[li]
    #                 # print(gb_single)
    #
    #                 # 取出value:粒球数据
    #                 gb = gb_dict_temp[li][0]
    #                 # 判断纯度是否满足要求，不满足则继续划分
    #                 label, p = get_label_and_purity(gb)
    #                 if p < purity:
    #                     gb_dict_new = splits_ball(gb_single).copy()
    #                     gb_dict.update(gb_dict_new)
    #                 else:
    #                     continue
    #             # print(gb_dict.keys())
    #             gb_len = len(gb_dict)
    #             ball_number_2 = len(gb_dict)
    #             # 粒球数和上一次划分的粒球数一样，即不再变化
    #             if ball_number_1 == ball_number_2:
    #                 break
    #
    #         return gb_dict
    #
    #     def splits_ball(gb_dict):
    #         # {center: [gb, distances]}
    #         center = []
    #         distances_other_class = []  # 粒球到异类点的距离
    #         balls = []  # 聚类后的label
    #         gb_dis_class = []  # 不同标签数据的距离
    #         center_other_class = []
    #         center_distances = []  # 新距离
    #         ball_list = {}  # 最后要返回的字典，键：中心点，值：粒球 + 到中心的距离
    #         distances_other = []
    #         distances_other_temp = []
    #
    #         centers_dict = []  # 中心list
    #         gbs_dict = []  # 粒球数据list
    #         distances_dict = []  # 距离list
    #
    #         # 取出字典中的数据:center,gb,distances
    #         # 取字典数据，包括键值
    #         gb_dict_temp = gb_dict.popitem()
    #         for center_split in gb_dict_temp[0].split('_'):
    #             center.append(float(center_split))
    #         center = np.array(center)  # 转为array
    #         centers_dict.append(center)  # 老中心加入中心list
    #         gb = gb_dict_temp[li][0]  # 取出粒球数据
    #         distances = gb_dict_temp[li][li]  # 取出到老中心的距离
    #         # print('center:', center)
    #         # print('gb:', gb)
    #         # print('distances:', distances)
    #
    #         # 分离不同标签数据的距离
    #         len_label = np.unique(gb[:, -2], axis=0)
    #         # print(len_label)
    #         for label in len_label.tolist():
    #             # 分离不同标签距离
    #             gb_dis_temp = []
    #             for i in range(0, len(distances)):
    #                 if gb[i, -2] == label:
    #                     gb_dis_temp.append(distances[i])
    #             if len(gb_dis_temp) > 0:
    #                 gb_dis_class.append(gb_dis_temp)
    #
    #         # 取新中心
    #         for i in range(0, len(gb_dis_class)):
    #             # print('gb_dis_class_i:', gb_dis_class[i])
    #
    #             # 最远异类点
    #             # center_other_temp = gb[distances.index(max(gb_dis_class[i]))]
    #
    #             # # 随机异类点
    #             ran = random.randint(0, len(gb_dis_class[i]) - li)
    #             center_other_temp = gb[distances.index(gb_dis_class[i][ran])]
    #             # print('center_other_temp:', center_other_temp)
    #
    #             if center[-2] != center_other_temp[-2]:
    #                 center_other_class.append(center_other_temp)
    #         # print('center_other_class:', center_other_class)
    #         centers_dict.extend(center_other_class)
    #         # print('centers_dict:', centers_dict)
    #
    #         distances_other_class.append(distances)
    #         # 计算到每个新中心的距离
    #         for center_other in center_other_class:
    #             balls = []  # 聚类后的label
    #             distances_other = []
    #             for feature in gb:
    #                 # 欧拉距离
    #                 distances_other.append(calculate_distances(feature[:-2], center_other[:-2]))
    #             # 新中心list
    #             # distances_dict.append(distances_other)
    #             distances_other_temp.append(distances_other)  # 临时存放到每个新中心的距离
    #             distances_other_class.append(distances_other)
    #         # print('distances_other_class:', len(distances_other_temp))
    #
    #         # 某一个数据到原中心和新中心的距离，取最小以分类
    #         for i in range(len(distances)):
    #             distances_temp = []
    #             distances_temp.append(distances[i])
    #             for distances_other in distances_other_temp:
    #                 distances_temp.append(distances_other[i])
    #             # print('distances_temp:', distances_temp)
    #             classification = distances_temp.index(min(distances_temp))  # 0:老中心；li,2...：新中心
    #             balls.append(classification)
    #         # 聚类情况
    #         balls_array = np.array(balls)
    #         # print("聚类情况：", balls_array)
    #
    #         # 根据聚类情况，分配数据
    #         for i in range(0, len(centers_dict)):
    #             gbs_dict.append(gb[balls_array == i, :])
    #         # print('gbs_dict:', len(gbs_dict))
    #
    #         # 分配新距离
    #         i = 0
    #         for j in range(len(centers_dict)):
    #             distances_dict.append([])
    #         # print('distances_dict:', distances_dict)
    #         for label in balls:
    #             distances_dict[label].append(distances_other_class[label][i])
    #             i += li
    #         # print('distances_dict:', distances_dict)
    #
    #         # 打包成字典
    #         for i in range(len(centers_dict)):
    #             gb_dict_key = str(float(centers_dict[i][0]))
    #             for j in range(li, len(centers_dict[i])):
    #                 gb_dict_key += '_' + str(float(centers_dict[i][j]))
    #             gb_dict_value = [gbs_dict[i], distances_dict[i]]  # 粒球 + 到中心的距离
    #             ball_list[gb_dict_key] = gb_dict_value
    #
    #         # print('ball_list:', ball_list.keys())
    #         return ball_list
    #
    #     data = self.data
    #     # print(data)
    #
    #     # 初始随机中心
    #     center_init = data[random.randint(0, len(data) - li), :]
    #     # center_init = data[:, li:3].mean(axis=0)
    #     # print(center_init)
    #
    #     distance_init = []
    #     for feature in data:
    #         # 初始中心距离
    #         distance_init.append(calculate_distances(feature[:-2], center_init[:-2]))
    #     # print('distance_init:', len(distance_init))
    #
    #     # 封装成字典
    #     gb_dict = {}
    #     gb_dict_key = str(center_init.tolist()[0])
    #     for i in range(li, len(center_init)):
    #         gb_dict_key += '_' + str(center_init.tolist()[i])
    #     gb_dict_value = [data, distance_init]
    #     gb_dict[gb_dict_key] = gb_dict_value
    #
    #     # 封装成字典
    #     gb_dict = {}
    #     gb_dict_key = str(center_init.tolist()[0])
    #     for i in range(li, len(center_init)):
    #         gb_dict_key += '_' + str(center_init.tolist()[i])
    #     gb_dict_value = [data, distance_init]
    #     gb_dict[gb_dict_key] = gb_dict_value
    #     # print("gb_dict:", gb_dict.keys())
    #
    #     print(purity_y)
    #
    #     # 分类划分
    #     gb_dict = splits(purity=purity_y, gb_dict=gb_dict)
    #     Clusterings = []
    #     for value in gb_dict.values():
    #         Cluster = GranularBall(value[0])
    #         Clusterings.append(Cluster)
    #     return Clusterings
    #
    #
    #     # print(gb_dict)
    #     # time.sleep(1000)
    #     #
    #     # ClusterLists = k_means(X=self.data_no_label, init=self.init_center, n_clusters=self.label_num)
    #     #
    #     #
    #     # data_label = ClusterLists[li]  # Get a list of tags
    #     # for i in range(self.label_num):
    #     #     Cluster_data = self.data[data_label == i, :]
    #     #     if len(Cluster_data) > li:
    #     #         Cluster = GranularBall(Cluster_data)
    #     #         Clusterings.append(Cluster)
    #     # return Clusterings

    def getBoundaryData(self):
        """
           Function function: get the points (boundary points) that need to be sampled in the pellet
       """
        if self.dim * 2 >= self.num:
            self.boundaryData = self.data
            return
        boundaryDataFalse = np.empty(shape=[0, self.dim])
        boundaryDataTrue = np.empty(shape=[0, self.dim + 2])
        for i in range(self.dim):
            centdataitem = np.tile(self.center, (1, 1))
            centdataitem[:, i] = centdataitem[:, i] + self.radius
            boundaryDataFalse = np.vstack((boundaryDataFalse, centdataitem))
            centdataitem = np.tile(self.center, (1, 1))
            centdataitem[:, i] = centdataitem[:, i] - self.radius
            boundaryDataFalse = np.vstack((boundaryDataFalse, centdataitem))
        list_path = []
        for boundaryDataItem in boundaryDataFalse:
            diffMat = np.tile(boundaryDataItem, (self.num, 1)) - self.data_no_label
            sqDiffMat = diffMat ** 2
            sqDistances = sqDiffMat.sum(axis=1)
            distances = sqDistances ** 0.5
            sortedDistances = distances.argsort()
            for i in range(self.num):
                if (self.data[sortedDistances[i]][-1] not in list_path and self.data[sortedDistances[i]][-2] == self.label):
                    boundaryDataTrue = np.vstack((boundaryDataTrue, self.data[sortedDistances[i]]))
                    list_path.append(self.data[sortedDistances[i]][-1])
                    break
        self.boundaryData = boundaryDataTrue

