#include <adolc/adolc.h>

int main(int, char **)
{
        double x = 3, y = 4;
        double a,b;

        a = fmax(x, y);
        b = fmin(x, y);

        return 0;
}
