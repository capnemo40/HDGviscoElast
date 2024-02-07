from netgen.geom2d import unit_square
from ngsolve import *
ngsglobals.msg_level = 1

'''
Convergence test for transient viscoelasticity. 
HDG approximation in terms of stress & velocity
Crank-Nicolson scheme

Unit square, manufactured solutions. 
p-conpvergence test (Figure 6.1)
Pure Dirichlet BCs
'''

def weaksym(k, h, dt, tend):
    
    # ********* Model coefficients and parameters ********* #

    t = Parameter(0.0)

    #relaxation time
    omega = 1

    #mass density
    rho = 1
    
    #Lamé coef. corresponding to C
    mu  = 1 
    lam = 3
    
    #Lamé coef. corresponding to D
    muV  = 2 
    lamV = 4
    
    muD  = muV - mu 
    lamD = lamV - lam
    
    #needed for A = C^{-1}
    a1E = 0.5 / mu
    a2E = lam / (4.0 * mu * (lam + mu))
    
    #needed for V = (D -C)^{-1}
    a1D = 0.5 / muD
    a2D = lamD / (4.0 * muD * (lamD + muD))
    

    # ********* Manufactured solutions for error analysis ********* #


    #Exact displacement: \bu
    exactu_0 = exp(-y)*cos(t)*sin(x)
    exactu_1 = exp(t + x)*cos(y)
    exactu = CoefficientFunction((exactu_0, exactu_1)) 
    
    #Exact velocity: \bv = \dot{\bu}
    exactv_0 = -exp(-y)*sin(t)*sin(x)
    exactv_1 = exp(t + x)*cos(y)
    exactv = CoefficientFunction((exactv_0, exactv_1)) 
    
    #Exact acceleration: \ba = \ddot{\bu}
    exacta_0 = -exp(-y)*cos(t)*sin(x)
    exacta_1 = exp(t + x)*cos(y)
    exacta = CoefficientFunction((exacta_0, exacta_1)) 
    
    #Strain tensor: \beps(\bu)
    epsu_00 = exp(-y)*cos(t)*cos(x)
    epsu_01 = (exp(t + x)*cos(y))/2 - (exp(-y)*cos(t)*sin(x))/2
    epsu_11 = -exp(t + x)*sin(y)
    epsu = CoefficientFunction((epsu_00, epsu_01, epsu_01, epsu_11), dims = (2,2)) 
    
    exactGam = 2*mu*epsu + lam*Trace(epsu)*Id(2)
    
    #Viscous component of stress: \dot{\bze} + \bze/\omega =  (\cD - \cC) \beps(\bv) 
    exactZe_00 = - exp(-t/omega)*((omega*exp(t/omega - y)*sin(t)*(lamD*cos(x) + 2*muD*cos(x)))/(omega**2 + 1) - (omega**2*exp(t/omega - y)*cos(t)*(lamD*cos(x) + 2*muD*cos(x)))/(omega**2 + 1) + (lamD*omega*exp(t + x + t/omega)*sin(y))/(omega + 1))
    exactZe_01 = exp(-t/omega)*((muD*omega*cos(t - x)*exp(t/omega)*exp(-y))/(2*(omega**2 + 1)) - (muD*omega**2*exp(t/omega)*exp(-y)*sin(t + x))/(2*(omega**2 + 1)) + (muD*omega**2*sin(t - x)*exp(t/omega)*exp(-y))/(2*(omega**2 + 1)) - (muD*omega*exp(t/omega)*exp(-y)*cos(t + x))/(2*(omega**2 + 1)) + (muD*omega*exp(t/omega)*exp(t)*exp(x)*cos(y))/(omega + 1))
    exactZe_11 = - exp(-t/omega)*((omega*exp(t + x + t/omega)*(lamD*sin(y) + 2*muD*sin(y)))/(omega + 1) - (lamD*omega**2*exp(t/omega - y)*cos(t + x))/(2*(omega**2 + 1)) + (lamD*omega*sin(t - x)*exp(t/omega - y))/(2*(omega**2 + 1)) - (lamD*omega**2*cos(t - x)*exp(t/omega - y))/(2*(omega**2 + 1)) + (lamD*omega*exp(t/omega - y)*sin(t + x))/(2*(omega**2 + 1)))
    exactZe = CoefficientFunction((exactZe_00, exactZe_01, exactZe_01, exactZe_11), dims = (2,2))
    
    #Total stress: \bsig = \bgam + \bze
    exactSig = exactGam + exactZe 
    
    #Source: \bF = \rho \ba - \bdiv \bsig
    divSigma1 = - exp(-t/omega)*((lamD*omega*exp((t + omega*t + omega*x)/omega)*sin(y))/(omega + 1) - (omega*exp((t - omega*y)/omega)*sin(t)*sin(x)*(lamD + 2*muD))/(omega**2 + 1) + (omega**2*exp((t - omega*y)/omega)*cos(t)*sin(x)*(lamD + 2*muD))/(omega**2 + 1)) - lam*(exp(t + x)*sin(y) + exp(-y)*cos(t)*sin(x)) - mu*exp(t + x)*sin(y) - mu*exp(-y)*cos(t)*sin(x) - (muD*omega*exp(-y)*(sin(t)*sin(x) + exp(t + x + y)*sin(y) + omega**2*exp(t + x + y)*sin(y) - omega*cos(t)*sin(x) + omega*sin(t)*sin(x) - omega**2*cos(t)*sin(x)))/(omega**3 + omega**2 + omega + 1)
    divSigma2 = (muD*omega*exp(-y)*(cos(x)*sin(t) + exp(t + x + y)*cos(y) + omega**2*exp(t + x + y)*cos(y) - omega*cos(t)*cos(x) + omega*cos(x)*sin(t) - omega**2*cos(t)*cos(x)))/(omega**3 + omega**2 + omega + 1) - mu*exp(t + x)*cos(y) - mu*exp(-y)*cos(t)*cos(x) - (omega*exp(-y)*(lamD*exp(t + x + y)*cos(y) - lamD*cos(x)*sin(t) + 2*muD*exp(t + x + y)*cos(y) + lamD*omega**2*cos(t)*cos(x) + lamD*omega**2*exp(t + x + y)*cos(y) + 2*muD*omega**2*exp(t + x + y)*cos(y) + lamD*omega*cos(t)*cos(x) - lamD*omega*cos(x)*sin(t)))/(omega**3 + omega**2 + omega + 1) - lam*(exp(t + x)*cos(y) + exp(-y)*cos(t)*cos(x))
    source = rho*exacta - CoefficientFunction( (divSigma1, divSigma2) )
    
    # ********* Mesh of the unit square ********* #

    mesh = Mesh(unit_square.GenerateMesh(maxh=h))

    # ********* Finite dimensional spaces ********* #

    S = L2(mesh, order =k)
    W = VectorL2(mesh, order =k+1)
    What = VectorFacetFESpace(mesh, order=k+1, dirichlet="bottom|left|right|top")
    fes = FESpace([S, S, S, S, S, S, W, What])
    
    # ********* test and trial functions for product space ****** #
    
    sigmaE1, sigmaE12, sigmaE2, sigmaV1, sigmaV12, sigmaV2, u, uhat = fes.TrialFunction()
    tauE1, tauE12, tauE2, tauV1, tauV12, tauV2, v, vhat = fes.TestFunction()
    
    sigmaE = CoefficientFunction(( sigmaE1, sigmaE12, sigmaE12, sigmaE2), dims = (2,2) )
    sigmaV  = CoefficientFunction(( sigmaV1,  sigmaV12,  sigmaV12,  sigmaV2),  dims = (2,2) )
    
    sigma = sigmaE + omega*sigmaV
    
    tauE   = CoefficientFunction(( tauE1,   tauE12,   tauE12,   tauE2),   dims = (2,2) )
    tauV   = CoefficientFunction(( tauV1,   tauV12,   tauV12,   tauV2),   dims = (2,2) )
    tau  = tauE + omega*tauV

    AsigmaE = a1E * sigmaE - a2E * Trace(sigmaE) *  Id(mesh.dim)
    VsigmaV  = a1D * sigmaV  - a2D * Trace(sigmaV)  *  Id(mesh.dim)
    
    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size 
  
    
    dS = dx(element_boundary=True)
    
    jump_u = u - uhat
    jump_v = v - vhat

    # ********* Bilinear forms ****** #

    a = BilinearForm(fes, condense=True)
    a += (1/dt)*rho*u*v*dx 
    a += (1/dt)*InnerProduct(AsigmaE, tauE)*dx + (1/dt)*InnerProduct(omega*VsigmaV, omega*tauV)*dx
    a +=   0.5*InnerProduct(VsigmaV, omega*tauV)*dx
    a +=   0.5*InnerProduct(sigma, grad(v))*dx    - 0.5*InnerProduct(grad(u), tau)*dx 
    a += - 0.5*InnerProduct( sigma*n, jump_v)*dS  + 0.5*InnerProduct(jump_u, tau*n)*dS 
    a +=   0.5*((k+1)**2/h)*jump_u*jump_v*dS 
    a.Assemble()

    inv_A = a.mat.Inverse(freedofs=fes.FreeDofs(coupling=True))
    
    
    M = BilinearForm(fes)
    M += (1/dt)*rho*u*v*dx 
    M += (1/dt)*InnerProduct(AsigmaE, tauE)*dx + (1/dt)*InnerProduct(omega*VsigmaV, omega*tauV)*dx
    M += - 0.5*InnerProduct(VsigmaV, omega*tauV)*dx
    M += - 0.5*InnerProduct(sigma, grad(v))*dx   + 0.5*InnerProduct(grad(u), tau)*dx
    M +=   0.5*InnerProduct( sigma*n, jump_v)*dS - 0.5*InnerProduct(jump_u, tau*n)*dS 
    M += - 0.5*((k+1)**2/h)*jump_u*jump_v*dS  
    M.Assemble()

    ft = LinearForm(fes)
    ft += source * v * dx

    # ********* instantiation of initial conditions ********* #    
    
    u0 = GridFunction(fes) 
    u0.components[0].Set(exactGam[0,0])
    u0.components[1].Set(exactGam[0,1])
    u0.components[2].Set(exactGam[1,1])
    u0.components[3].Set(exactZe[0,0]/omega)
    u0.components[4].Set(exactZe[0,1]/omega)
    u0.components[5].Set(exactZe[1,1]/omega)
    u0.components[6].Set(exactv)
    u0.components[7].Set(exactv, dual=True)

    
    ft.Assemble()
    
    res = u0.vec.CreateVector()
    b0  = u0.vec.CreateVector()
    b1  = u0.vec.CreateVector()
    
    b0.data = ft.vec

    t_intermediate = dt # time counter within one block-run
    
    # ********* Time loop ************* #

    while t_intermediate < tend:

        t.Set(t_intermediate)
        ft.Assemble()
        b1.data = ft.vec
     
        res.data = M.mat*u0.vec + 0.5*(b0.data + b1.data)

        u0.vec[:] = 0.0 
        u0.components[7].Set(exactv, BND, dual=True)#BDN ---> Frontera Dirichlet

        res.data = res - a.mat * u0.vec

        res.data += a.harmonic_extension_trans * res

        u0.vec.data += inv_A * res
        
        u0.vec.data += a.inner_solve * res
        u0.vec.data += a.harmonic_extension * u0.vec
        
        b0.data = b1.data
        t_intermediate += dt
        
        print('t=%g' % t_intermediate)
        
        #Redraw(blocking=True)

    gfsigmaE1, gfsigmaE12, gfsigmaE2, gfsigmaV1, gfsigmaV12, gfsigmaV2, gfu = u0.components[0:7]

    gfsigmaE = CoefficientFunction(( gfsigmaE1, gfsigmaE12, gfsigmaE12, gfsigmaE2), dims = (2,2) )
    gfsigmaV  = CoefficientFunction(( gfsigmaV1,  gfsigmaV12,  gfsigmaV12,  gfsigmaV2),  dims = (2,2) )
    gfsigma = gfsigmaE + omega*gfsigmaV



    norm_u= (gfu - exactv) * (gfu - exactv)
    norm_u = Integrate(norm_u, mesh)
    norm_u = sqrt(norm_u)
    #print(norm_r)

    norm_s  = InnerProduct(a1E*(exactGam - gfsigmaE) - a2E*Trace(exactGam - gfsigmaE)*  Id(mesh.dim), exactGam - gfsigmaE)
    norm_s += InnerProduct(a1D*(exactZe  - gfsigmaV) - a2D*Trace(exactZe  - gfsigmaV)*  Id(mesh.dim), exactZe - gfsigmaV)
    norm_s = Integrate(norm_s, mesh)
    norm_s = sqrt(norm_s)
    #print(norm_s)

    return norm_s, norm_u

# ********* Convergence table ************* # 

def hconvergenctauEble(e_1, e_2, maxk):
    print("============================================================")
    print(" k   Errors_s   Error_u   ")
    print("------------------------------------------------------------")
    
    for i in range(maxk):
        print(" %-4d %8.2e    %8.2e   " % (i, e_1[i], 
               e_2[i]))

    print("============================================================")

# ********* Error collector ************* # 

def collecterrors(maxk, h, dt, tend):
    l2e_s = []
    l2e_r = []
    for k in range(0, maxk):
        er_1, er_2 = weaksym(k, h, dt, tend)
        l2e_s.append(er_1)
        l2e_r.append(er_2)
    return l2e_s, l2e_r


# ============= MAIN DRIVER ==============================

maxk = 6 #number of k refinements
dt = 10e-6
tend = 0.5
h = 1/4

er_s, er_u = collecterrors(maxk, h, dt, tend)
hconvergenctauEble(er_s, er_u, maxk)
