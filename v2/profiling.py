import numpy as np
import alphax_utils as apy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import timeit


def test_something():
	srcfile = "/home/rkarim/Research/AlphaX/PyAlpha_drafting/test_data/Ssets/s1.txt"
	points = np.genfromtxt(srcfile)
	return points
data = test_something()
apy.initialize(data)
apy.KEY.alpha_step = 0.95
apy.KEY.orphan_tolerance = 80
apy.recurse()

