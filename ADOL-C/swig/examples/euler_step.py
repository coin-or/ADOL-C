from __future__ import print_function
import adolc
import numpy as np

steps = 100
h = 0.01

def euler_step_act(n,yin,m,yout):
    yout[0] = yin[0] + h*yin[0]
    yout[1] = yin[1] + h*2*yin[1]

class euler_step_edf(adolc.EDFobject):
    def __init__(self,tag):
        super(self.__class__,self).__init__()
        self.tag_ext_fct = tag
    def function(self,arg):
        (n,yin,m,yout) = arg
        yout[0] = yin[0] + h*yin[0]
        yout[1] = yin[1] + h*2*yin[0]
        return 1
    def zos_forward(self,arg):
        (n,yin,m,yout) = arg
        adolc.set_nested_ctx(self.tag_ext_fct,1)
        ny = adolc.zos_forward(self.tag_ext_fct,2,2,0,yin)
        adolc.set_nested_ctx(self.tag_ext_fct,0)
        np.copyto(yout,ny)
        return 1
    def fos_forward(self,arg):
        (n,yin,dyin,m,yout,dyout) = arg
        adolc.set_nested_ctx(self.tag_ext_fct,1)
        ny,ndy = adolc.fos_forward(self.tag_ext_fct,2,2,0,yin,dyin)
        adolc.set_nested_ctx(self.tag_ext_fct,0)
        np.copyto(yout,ny)
        np.copyto(dyout,ndy)
        return 1
    def fov_forward(self,arg):
        (n,yin,p,dyin,m,yout,dyout) = arg
        adolc.set_nested_ctx(self.tag_ext_fct,1)
        ny,ndy = adolc.fov_forward(self.tag_ext_fct,2,2,p,yin,dyin)
        adolc.set_nested_ctx(self.tag_ext_fct,0)
        np.copyto(yout,ny)
        np.copyto(dyout,ndy)
        return 1
    def fos_reverse(self,arg):
        (m,u,n,z,x,y) = arg
        adolc.set_nested_ctx(self.tag_ext_fct,1)
        ny = adolc.zos_forward(self.tag_ext_fct,2,2,1,x)
        nz = adolc.fos_reverse(self.tag_ext_fct,2,2,u)
        np.copyto(z,nz)
        return 1
    def fov_reverse(self,arg):
        (m,q,u,n,z,x,y) = arg
        adolc.set_nested_ctx(self.tag_ext_fct,1)
        ny = adolc.zos_forward(self.tag_ext_fct,2,2,1,x)
        nz = adolc.fov_reverse(self.tag_ext_fct,2,2,q,u)
        np.copyto(z,nz)
        return 1


if __name__ == '__main__':
    tag_full = 1
    tag_part = 2
    tag_ext_fct = 3
    n = 2
    m = 2
    t0 = 0.0
    tf = 1.0
    conp = [ 1.0, 1.0 ]
    adolc.trace_on(tag_full)
    con = adolc.as_adouble(conp)
    for c in con:
        c.declareIndependent()
    y = con
    ynew = adolc.as_adouble([ 0.0, 0.0 ])
    for i in range(steps):
        euler_step_act(n,y,m,ynew)
        y = ynew
    f = y[0] + y[1]
    f.declareDependent()
    adolc.trace_off()
    grad = adolc.gradient(tag_full,conp)
    print(" full taping:\n gradient=(",grad[0],", ",grad[1],")\n\n")

    # now taping external function
    adolc.trace_on(tag_ext_fct)
    con = adolc.as_adouble(conp)
    for c in con:
        c.declareIndependent()
    y = con
    ynew = adolc.as_adouble([ 0.0, 0.0 ])
    euler_step_act(2,y,2,ynew)
    for c in ynew:
        c.declareDependent()
    adolc.trace_off()

    edf = euler_step_edf(tag_ext_fct)    
    # now use the external function
    adolc.trace_on(tag_part)
    con = adolc.advector(conp)
    for c in con:
        c.declareIndependent()
    y = con
    ynew = adolc.advector([0.0,0.0])
    for i in range(steps):
        edf.call(2,y,2,ynew)
        y = ynew
    f = y[0] + y[1]
    f.declareDependent()
    adolc.trace_off()
    gradp = adolc.gradient(tag_part,conp)
    print(" taping with external function facility:\n gradient=(",gradp[0],", ",gradp[1],")\n\n")
