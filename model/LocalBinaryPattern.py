import numpy as np
from skimage.feature import local_binary_pattern


class LocalBinaryPattern:
    def __init__(self, num_points, radius):
        self.numPoints = num_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist
