import unittest

from knn import KNearestNeighbour 

class TestKNN(unittest.TestCase):
    """
        peerforms unit test for class KNearestNeighbour
    """
    def test_calculate_distance(self):
        """

        """
        point = (0,4)
        test_point = (0,1)
        dist = KNearestNeighbour.calculate_distance(point=point, test_point=test_point)
        self.assertEqual(dist, 3)



if __name__ == "__main__":
    unittest.main()