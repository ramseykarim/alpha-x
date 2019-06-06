import timeit
import numpy as np
import matplotlib.pyplot as plt
import alphax_utils as utils
from scipy.spatial import Delaunay
from collections import deque


def setup_fn(npts, ndim):
    a = np.random.uniform(size=(npts, ndim))
    tri_a=Delaunay(a)
    sx=a[tri_a.simplices]
    return sx

def run_fn(sx):
    utils.cayley_menger_vr(sx)

def run_fn_slow(sx):
    for t in sx:
        utils.OLD_cayley_menger_vr(t)

def time_OLD_vs_NEW_volume_cr(npts, ndim):
    setupcode1 = "from __main__ import setup_fn, run_fn; sx=setup_fn({:d}, {:d})".format(npts, ndim)
    stmt1 = "run_fn(sx)"
    setupcode2 = "from __main__ import setup_fn, run_fn_slow; sx=setup_fn({:d}, {:d})".format(npts, ndim)
    stmt2 = "run_fn_slow(sx)"
    rpt = 5
    print("NEW")
    tnew = timeit.timeit(stmt=stmt1, setup=setupcode1, number=rpt)*1000/float(rpt)
    print("{:6.2f} microseconds".format(tnew))
    print("OLD")
    told = timeit.timeit(stmt=stmt2, setup=setupcode2, number=rpt)*1000/float(rpt)
    print("{:6.2f} microseconds".format(told))
    return tnew, told

def profile_CMVR_ndimensions():
    tolds, tnews = [], []
    ndims = list(range(2, 7))
    for ndim in ndims:
        tnew, told = time_OLD_vs_NEW_volume_cr(30, ndim)
        tolds.append(told)
        tnews.append(tnew)
    ndims = np.array(ndims)
    fit_old = np.polyfit(ndims, np.log(tolds), deg=1)
    fit_new = np.polyfit(ndims, np.log(tnews), deg=1)
    mo, bo = fit_old[0], np.exp(fit_old[1])
    mn, bn = fit_new[0], np.exp(fit_new[1])
    plt.plot(ndims, bo * np.exp(ndims*mo), '--', color='k')
    plt.plot(ndims, bn * np.exp(ndims*mn), '--', color='k')
    print("OLD: ", mo, bo)
    print("NEW: ", mn, bn)
    plt.plot(ndims, tolds, '.', label="OLD", markersize=15)
    plt.plot(ndims, tnews, '.', label="NEW", markersize=15)
    plt.yscale('log')
    plt.xlabel("number of dimensions")
    plt.ylabel("runtime (log[ms])")
    plt.legend()
    plt.show()

def profile_CMVR_npoints():
    tolds, tnews = [], []
    npts = list(map(int, 10**np.arange(1, 3., 0.3)))
    for npt in npts:
        print(np.log10(npt))
        tnew, told = time_OLD_vs_NEW_volume_cr(npt, 3)
        tolds.append(told)
        tnews.append(tnew)
    npts = np.array(npts)
    plt.plot(npts, tolds, '.', label="OLD", markersize=15)
    plt.plot(npts, tnews, '.', label="NEW", markersize=15)
    fit_old = np.polyfit(np.log(npts), np.log(tolds), deg=1)
    fit_new = np.polyfit(np.log(npts), np.log(tnews), deg=1)
    mo, bo = fit_old[0], np.exp(fit_old[1])
    mn, bn = fit_new[0], np.exp(fit_new[1])
    print("OLD: ", mo, bo)
    print("NEW: ", mn, bn)
    plt.plot(npts, bo * npts**mo, '--', color='k')
    plt.plot(npts, bn * npts**mn, '--', color='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("number of points")
    plt.ylabel("runtime (log[ms])")
    plt.legend()
    plt.show()


def run_deque():
    return deque()
def run_list():
    return []
def add_item(a):
    a.append(True)
def profile_list_vs_deque():
    rpt = int(1e8)
    setupcode_deque = "from __main__ import run_deque; a = run_deque()"
    setupcode_list = "from __main__ import run_list; a = run_list()"
    tdeque = timeit.timeit(stmt="a.append(100.001)", setup=setupcode_deque, number=rpt)
    tlist = timeit.timeit(stmt="a.append(100.001)", setup=setupcode_list, number=rpt)
    print("List : {:6.2f}".format(tlist))
    print("Deque: {:6.2f}".format(tdeque))
    # LIST IS ALMOST ALWAYS FASTER! deque has ~2x slower build

profile_list_vs_deque()
