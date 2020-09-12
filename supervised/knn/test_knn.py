import unittest

from knn import KNearestNeighbour 

class TestKNN(unittest.TestCase):
    """
        performs unit test for class KNearestNeighbour
    """
    def test_calculate_distance(self):
        """

        """
        point = 0,1
        test_point = 0,2
        dist = KNearestNeighbour.calculate_distance(self, point=point, test_point=test_point)
        self.assertEqual(dist, 1)

    def test_predict(self):
        """
            performs unit test for predict
        """
        knn = KNearestNeighbour(2)
        knn.fit([1, 2, 3769, 67, 75, 7, 5 , 6])
        knn.predict(5)
        res = len(KNearestNeighbour.predict(self, 5))
        self.assertEqual(res, 5)
        pass


if __name__ == "__main__":
    unittest.main()