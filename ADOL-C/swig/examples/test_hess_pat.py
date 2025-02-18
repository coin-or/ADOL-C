import adolc
import numpy as np

def tapeFunction(x, rowno):
   adolc.trace_on(rowno)
   ax = adolc.as_adouble(x)

   # marking the x variables as independent
   for item in iter(ax):
      item.declareIndependent()

   L   = ax[0]
   Inp = ax[1]
   Out = ax[2]
   P   = ax[3]

   ay = adolc.as_adouble(0)
   if rowno == 0:
      ay = ax[3] * ax[2];
   elif rowno == 1:
      hold1 = (pow(ax[0],(-1.0)) + pow(ax[1],(-1.0)))
      hold2 = pow(hold1,(-1.0))

      ay = hold2

   ay.declareDependent()
   adolc.trace_off()


def initialiseAutoDiff():
   x = [0.5, 0.5, 0.0, 0.0]
   tapeFunction(x, 0)
   tapeFunction(x, 1)


def computeHessianStructure():
   x = [0.5, 0.5, 0.0, 0.0]
   hesspat = adolc.hess_pat(0, x, 0)
   assert(np.array_equal(hesspat[0], [0, 0, 1, 1]))
   assert(np.array_equal(hesspat[1], [3, 2]))

   hesspat = adolc.hess_pat(1, x, 0)
   assert(np.array_equal(hesspat[0], [2, 2, 0, 0]))
   assert(np.array_equal(hesspat[1], [0, 1, 0, 1]))

if __name__ == "__main__":
   initialiseAutoDiff()
   computeHessianStructure()
