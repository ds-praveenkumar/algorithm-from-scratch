# knn.py
import numpy as np

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
        self.data = None

    def calculate_distance(method="euclidean", point=(0,0), test_point=(0,0)):
        """
            distance =  
        """
        if method == "euclidean":
            sub1, sub2 = np.subtract(point[0], test_point[0]), np.subtract(point[1], test_point[1])
            add = np.add(sub1**2,sub2**2)
            dist = np.sqrt(add)
            return dist




    def fit(self, data=list):
        """
            This trains the data 
        """
        self.data = data
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

        
    def predict(self, test_data=list):
        """

        """
        

if __name__ == "__main__":
    knn = KNearestNeighbour(2)
    knn.fit([1,2,3769,67, 75])