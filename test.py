#coding=utf8
from utils import *


if __name__ == '__main__':
    a = np.array([0,0,0,1,1,1,0])
    b = np.array([1,1,1,0,0,0,0])
    print calc_purity(a,b,2,2)
    print calc_inversepurity(a,b,2,2)
    print calc_fmeasureij(a,b,0,1)
    print calc_fmeasureij(a,b,0,0)
    print calc_fmeasureij(a,b,1,0)
    print calc_fmeasureij(a,b,1,1)
    print calc_fmeasure(a,b,2,2)
    