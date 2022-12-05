from tslearn.metrics import dtw
from tslearn.metrics import soft_dtw
import numpy
from math import sqrt


def dtw_distance(first_timeseries, second_timeseries):
    return dtw(first_timeseries, second_timeseries)


def soft_dtw_distance(first_timeseries, second_timeseries):
    return soft_dtw(first_timeseries)


def euclidean_distance(first_timeseries, second_timeseries):
    s3 = [(a - b) ** 2 for a, b in zip(first_timeseries, second_timeseries)]
    dist = sqrt(numpy.sum(s3))
    return dist
