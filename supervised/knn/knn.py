# knn.py
import numpy as np
import operator

class KNearestNeighbour:
    """
        This class implements k Nearest Neighbour
    """
    def __init__(self, n_neighbours=2):
        """
            initializes k
        """
        self.n_neighbours = n_neighbours
        self.data_points_len = None
        self.data_set = None

    def calculate_distance(self, point=tuple, test_point=tuple, method="euclidean" ):
        """
            distance =  sqrt( (x1-y1)^2 + (x2-y2)^2 )
        """
        dist = None
        if method == "euclidean":
            sub1, sub2 = np.subtract(point[0], test_point[0]), np.subtract(point[1], test_point[1])
            squared = np.add(sub1**2, sub2**2)
            dist = np.sqrt(squared)
            return dist


    def fit(self, data=list):
        """
            This fits the data to the data to the algorithm 
        """
        self.data_set = data.copy()
        self.data_points_len = len(data)
        if self.n_neighbours > self.data_points_len:
            raise Exception(f"n_neighbours greater than number of data points: found clusters:{self.n_neighbours} data points length {self.data_points_len}'")
        else:
            pass

        random_k = {}
        for index in range(1,self.n_neighbours+1):
            _random_k = np.random.choice(np.array(data))
            data.remove(_random_k)
            random_k[index] = (_random_k)

        
    def predict(self, test_point):
        """
            This method takes the test data point and predicts the most frequent class
        """
        dist_list = []
        for val in self.data_set:
            distance = self.calculate_distance((0,val), (0,test_point))  
            dist_list.append((val, distance)) 
            dist_list.sort(key=operator.itemgetter(1))
        return dist_list[0][0] 

if __name__ == "__main__":
    knn = KNearestNeighbour(2)
    knn.fit([1, 2, 3769, 67, 75, 7, 5 , 6, 7, 8])
    test_point = 700767
    print(f"test label:{test_point}, predicted label: {knn.predict(test_point)}")