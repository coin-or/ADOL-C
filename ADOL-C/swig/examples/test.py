from __future__ import print_function
import numpy as np
from adolc import *

def do_taping(tag,x):
    trace_on(tag)
    ax = as_adouble(x)
    for item in iter(ax):
       item.declareIndependent()
    if (ax[0] < 0):
       ay = ax[2]*ax[1]*2.0
    else:
       ay = ax[2]+3.0*ax[1]
    ay.declareDependent()
    trace_off()


if __name__ == '__main__':
    x = [ 1, 2, 3 ]
    disableBranchSwitchWarnings()
    do_taping(1,x)
    g = gradient(1,x)
    print(g)
    x = [ -1, 2, 3]
    try:
       g = gradient(1,x)
    except BranchException: 
       do_taping(1,x)
       g = gradient(1,x)
    print(g)
    
