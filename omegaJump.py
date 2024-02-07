from ngsolve import *
from netgen.geom2d import SplineGeometry

'''
Test for transient elastic/viscoelastic wave propagation. 
HDG approximation in terms of stress & velocity
Crank-Nicolson scheme

Rectangular domain, left_domain = viscoelastic, right_domain = elastic. 
Simulation corresponding to Figure 6.5
'''

# ********* Geometry and mesh ********* #

geo = SplineGeometry()
L = 8
pnts = [ (-L,-L), (0,-L), (L,-L), (L,L), (0,L), (-L,L) ]
pind = [ geo.AppendPoint(*pnt) for pnt in pnts ]

geo.Append(['line',pind[0],pind[1]],leftdomain=1,rightdomain=0,bc="bottom")
geo.Append(['line',pind[1],pind[2]],leftdomain=2,rightdomain=0,bc="bottom")
geo.Append(['line',pind[2],pind[3]],leftdomain=2,rightdomain=0,bc="right")
geo.Append(['line',pind[3],pind[4]],leftdomain=2,rightdomain=0,bc="top")
geo.Append(['line',pind[4],pind[5]],leftdomain=1,rightdomain=0,bc="top")
geo.Append(['line',pind[5],pind[0]],leftdomain=1,rightdomain=0,bc="left")
geo.Append(['line',pind[1],pind[4]],leftdomain=1,rightdomain=2,bc="interface")

geo.SetMaterial(1, "left_dom")
geo.SetMaterial(2, "right_dom")

h = 0.25
k = 2
mesh = Mesh( geo.GenerateMesh(maxh=h) )


# ********* Model coefficients and parameters ********* #

dt = 0.01# time step size ∆t

#Draw(mesh)

t = Parameter(0.0)
tend = 2 #choose: 0.0, 1, 1.5, 2

#relaxation time (left domain)
omega = 0.7

#mass density
rho_left = 1
rho_right = 1
    
#Lamé coef. corresponding to C
mu_left = 2 #1.43**2
lam_left = 3.5 #2.74**2 - 2*mu_left

mu_right  = 2 #1.43**2
lam_right = 3.5 #2.74**2 - 2*mu_left

#Lamé coef. corresponding to D
muV = 3 #(1.01470*(1.43**2))/omega # (left domain)
lamV = 5# (1.0133*(2.74**2))/omega - 2*muV #left (left domain)

#Lamé coef. corresponding to D-C
    
lamD = lamV - lam_left #(left domain)
muD = muV - mu_left #(left domain)

domain_values_rho = {'left_dom': rho_left,  'right_dom': rho_right}
values_list_rho = [domain_values_rho[mat]
               for mat in mesh.GetMaterials()]
rho = CoefficientFunction(values_list_rho)

domain_values_mu = {'left_dom': mu_left,  'right_dom': mu_right}
values_list_mu = [domain_values_mu[mat]
               for mat in mesh.GetMaterials()]
mu = CoefficientFunction(values_list_mu)

domain_values_lam = {'left_dom': lam_left,  'right_dom': lam_right}
values_list_lam = [domain_values_lam[mat]
               for mat in mesh.GetMaterials()]
lam = CoefficientFunction(values_list_lam)


#needed for A = C^{-1}
domain_values_a1E = {'left_dom': 0.5 / mu_left,  'right_dom': 0.5 / mu_right}
values_list_a1E = [domain_values_a1E[mat]
               for mat in mesh.GetMaterials()]
a1E = CoefficientFunction(values_list_a1E)

domain_values_a2E = {'left_dom': lam_left / (4.0 * mu_left * (lam_left + mu_left)),  'right_dom': lam_right / (4.0 * mu_right * (lam_right + mu_right))}
values_list_a2E = [domain_values_a2E[mat]
               for mat in mesh.GetMaterials()]
a2E = CoefficientFunction(values_list_a2E)


#needed for V = (D -C)^{-1}

a1D = 0.5 / muD #(left domain)

a2D = lamD / (4.0 * muD * (lamD + muD)) #(left domain)
    
# ********* Source term ********* #

rad = CoefficientFunction(sqrt(x**2 + y**2))
ge = (1 - rad**2/(5*h)**2)*CoefficientFunction((x/rad, y/rad))*IfPos( 5*h - rad, 1.0, 0.0 )

fre = 15#1.43/h
t00 = 1/fre

source =  -2*pi**2 * fre**2 *(t-t00)*exp(-pi**2 * fre**2 * (t - t00)**2)*ge*IfPos(2*t00 - t, 1.0, 0.0)

 # ********* Finite dimensional spaces ********* #

S  = L2(mesh, order =k)
SV = L2(mesh, order =k, definedon = "left_dom")
W = VectorL2(mesh, order =k+1)
What = VectorFacetFESpace(mesh, order=k+1, dirichlet="top, bottom, right, left")
fes = FESpace([S, S, S, SV, SV, SV, W, What])

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
a += (1/dt)*InnerProduct(AsigmaE, tauE)*dx + (1/dt)*InnerProduct(omega*VsigmaV, omega*tauV)*dx("left_dom")
a +=   0.5*InnerProduct(VsigmaV, omega*tauV)*dx("left_dom")
a +=   0.5*InnerProduct(sigma, grad(v))*dx    - 0.5*InnerProduct(grad(u), tau)*dx 
a += - 0.5*InnerProduct( sigma*n, jump_v)*dS  + 0.5*InnerProduct(jump_u, tau*n)*dS 
a +=   0.5*((k+1)**2/h)*jump_u*jump_v*dS 
a.Assemble()

inv_A = a.mat.Inverse(freedofs=fes.FreeDofs(coupling=True))
        
        
M = BilinearForm(fes)
M += (1/dt)*rho*u*v*dx 
M += (1/dt)*InnerProduct(AsigmaE, tauE)*dx + (1/dt)*InnerProduct(omega*VsigmaV, omega*tauV)*dx("left_dom")
M += - 0.5*InnerProduct(VsigmaV, omega*tauV)*dx("left_dom")
M += - 0.5*InnerProduct(sigma, grad(v))*dx   + 0.5*InnerProduct(grad(u), tau)*dx
M +=   0.5*InnerProduct( sigma*n, jump_v)*dS - 0.5*InnerProduct(jump_u, tau*n)*dS 
M += - 0.5*((k+1)**2/h)*jump_u*jump_v*dS  
M.Assemble()

ft = LinearForm(fes)
ft += source * v * dx

# ********* instantiation of initial conditions ****** #

disp0 = GridFunction(W) 
disp0.vec[:] = 0.0

disp = GridFunction(W) 
disp.vec[:] = 0.0
Draw(100*sqrt(disp[0]**2 + disp[1]**2), mesh, "displacement", autoscale=False, min=0, max=1)

u0 = GridFunction(fes) 
u0.vec[:] = 0.0

u1 = GridFunction(fes) 
u1.vec[:] = 0.0
    
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

    u1.vec[:] = 0.0 

    res.data = res - a.mat * u1.vec
    res.data += a.harmonic_extension_trans * res
    u1.vec.data += inv_A * res 
    u1.vec.data += a.inner_solve * res
    u1.vec.data += a.harmonic_extension * u1.vec
    
    disp.vec.data = disp0.vec + 0.5*dt*(u1.components[6].vec + u0.components[6].vec)
    
    u0.vec.data = u1.vec
    disp0.vec.data = disp.vec
        
    b0.data = b1.data
    t_intermediate += dt
        
    print('t=%g' % t_intermediate)
        
    Redraw(blocking=True)
    

vtk = VTKOutput(ma=mesh,coefs=[disp], names = ["displacement"], filename="result_dos", subdivision=3)
# Exporting the results:
vtk.Do()
