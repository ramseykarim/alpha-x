import numpy as np
import utils2_draft as apy

def get_data():
    data = np.genfromtxt("../../PyAlpha_drafting/test_data/uniform600_gap_hiSN.txt", skip_header=1)  # GT = 15
    return data

data = get_data()[:40, :]
apy.initialize(data)
apy.recurse()
