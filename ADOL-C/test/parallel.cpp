#include <adolc/adouble.h>            // use of active doubles
#include <adolc/drivers/drivers.h>    // use of "Easy to Use" drivers
// gradient(.) and hessian(.)
#include <adolc/taping.h>             // use of taping

#include <iostream>
using namespace std;

#include <cstdlib>
#include <thread>

void derive(const short my_tape, double init)
{
  const int n = 100;

  double *xp = new double[n];
  double  yp = 0.0;
  adouble *x = new adouble[n];        
  adouble  y = 1;

  for(int i = 0; i < n; i++)
  {
    xp[i] = (i + init)/(2. + i);           // some initialization
  }

  trace_on(my_tape);
  
  for(int i = 0; i < n; i++) {
    x[i] <<= xp[i];
    y *= x[i];
  }

  y >>= yp;

  delete[] x;
  
  trace_off();

  double* g = new double[n];
  
  gradient(my_tape, n, xp, g);

  double** H = new double*[n];
  
  for(int i = 0; i < n; i++)
  {
    H[i] = (double*) new double[i + 1];
  }
  
  hessian(my_tape, n, xp, H);

  double errh = 0.;

  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < n; j++)
    {
      if (i>j)
      {
        errh += fabs(H[i][j] - g[i]/xp[j]);
      }
    } // end for
  } // end for

  std::cout << "Computed Hessian in tape " << my_tape
            << ", error = "
            << errh
            << std::endl;

  for(int i = 0; i < n; ++i)
  {
    delete[] H[i];
  }
  
  delete[] H;
  delete[] g;
}


int main()
{
  std::vector<std::thread> threads;

  for(int i = 1; i <= 10; ++i)
  {
    threads.push_back(std::thread(derive, i, (double) i));
  }

  for(auto& thread: threads)
  {
    thread.join();
  }
    
  return 0;
}

