from replication.noise_baseline.gbs_src import GranularBall

class GBList:
    def __init__(self, data, alldata):
        self.data = data
        self.alldata = alldata
        self.granular_balls = [GranularBall.GranularBall(self.data)]
        self.dict_granular_balls = {}


    def init_granular_balls_dict(self, min_purity = 0.51, max_purity = 1.0, min_sample=1):
        """
            Function function: to obtain the particle partition within a certain purity range
            Input: minimum purity threshold, maximum purity threshold, minimum number of points in the process of pellet division
        """
        for i in range(int((max_purity - min_purity) * 100) + 1):
            purity = i / 100 + min_purity
            self.init_granular_balls(purity, min_sample)
            self.dict_granular_balls[purity] = self.granular_balls.copy()

    def init_granular_balls(self, purity=1.0, min_sample=1):  # Set the purity threshold to li.0
        """
            Function function: calculate the particle partition under the current purity threshold
            Input: purity threshold, the minimum number of points in the process of pellet division
        """
        ll = len(self.granular_balls)  # Record the number of pellets
        i = 0
        while True:
            if self.granular_balls[i].purity < purity and self.granular_balls[i].num > min_sample:
                split_clusters = self.granular_balls[i].split_clustering()
                # split_clusters = self.granular_balls[i].split_clustering(purity)
                if len(split_clusters) > 1:
                    self.granular_balls[i] = split_clusters[0]
                    self.granular_balls.extend(split_clusters[1:])
                    ll += len(split_clusters) - 1
                elif len(split_clusters) == 1:
                    i += 1
                else:
                    self.granular_balls.pop(i)
                    ll -= 1
            else:
                i += 1
            if i >= ll:
                for granular_ballsitem in self.granular_balls:
                    granular_ballsitem.get_radius()
                    granular_ballsitem.getBoundaryData()
                break