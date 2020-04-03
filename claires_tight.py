import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import sys
from decimal import Decimal
import math
import yaml
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

open_param_file = open('parameter_files/parameters.yaml')
param = yaml.load(open_param_file)
eps_b = np.sqrt(param['n_b'])
c = param['constants']['c']
hbar_eVs = param['constants']['hbar_eVs']
e = param['constants']['e']
prec = param['precision']

inputs = np.loadtxt(param['inputs'],skiprows=1)

rod_centers = inputs[:,0:2]*1E-7
dip_centers = inputs[:,2:4]*1E-7 # the center of the rod is not necessarily the center of the dipole
L_vecs = inputs[:,4:6]*1E-7
S_vecs = inputs[:,6:8]*1E-7

numSph = len(inputs)
numIndModes = (inputs.shape[1]-4)/2

rL_tb = 70E-7; rS_tb = 121E-7; #top bottom
rL_lr = 85E-7;  rS_lr = 70E-7; #left right

wp = 9.
eps_inf= 9.
w0 = np.sqrt(wp**2/((eps_inf-1)+3))
w0_Ltb = w0; w0_Llr = w0; w0_Stb = w0; w0_Slr = w0

VL_tb = 4/3*np.pi*rL_tb**3
VS_tb = 4/3*np.pi*rS_tb**3

mL_tb = 4*np.pi*e**2*((9. - 1)+3)/(9*(w0_Ltb/hbar_eVs)**2*VL_tb)
mS_tb = 4*np.pi*e**2*((9. - 1)+3)/(9*(w0_Stb/hbar_eVs)**2*VS_tb)

VL_lr = 4/3*np.pi*rL_lr**3
VS_lr = 4/3*np.pi*rS_lr**3

mL_lr = 4*np.pi*e**2*((9. - 1)+3)/(9*(w0_Llr/hbar_eVs)**2*VL_lr)
mS_lr = 4*np.pi*e**2*((9. - 1)+3)/(9*(w0_Slr/hbar_eVs)**2*VS_lr)


w0QS_L = np.array([w0_Llr, w0_Llr, w0_Ltb, w0_Ltb ])
w0QS_S = np.array([w0_Slr, w0_Slr, w0_Stb, w0_Stb ])

mQS_L = np.array([mL_lr, mL_lr, mL_tb, mL_tb])
mQS_S = np.array([mS_lr, mS_lr, mS_tb, mS_tb])

#############################################################
### long wavelength approximation ###
#############################################################

m_L =  np.array([mL_lr + e**2/(rL_lr*c**2) , mL_lr + e**2/(rL_lr*c**2), mL_tb + e**2/(rL_tb*c**2), mL_tb + e**2/(rL_tb*c**2)])
m_S =  np.array([mS_lr + e**2/(rS_lr*c**2) , mS_lr + e**2/(rS_lr*c**2), mS_tb + e**2/(rS_tb*c**2), mS_tb + e**2/(rS_tb*c**2)])

w0_L = np.array([w0_Llr*np.sqrt(mQS_L[0]/m_L[0]), w0_Llr*np.sqrt(mQS_L[1]/m_L[1]), w0_Ltb*np.sqrt(mQS_L[2]/m_L[2]), w0_Ltb*np.sqrt(mQS_L[3]/m_L[3]) ])
w0_S = np.array([w0_Slr*np.sqrt(mQS_S[0]/m_S[0]), w0_Slr*np.sqrt(mQS_S[1]/m_S[1]), w0_Stb*np.sqrt(mQS_S[2]/m_S[2]), w0_Stb*np.sqrt(mQS_S[3]/m_S[3]) ])

gamNR_L = 0.07 * np.array([mQS_L[0]/m_L[0], mQS_L[1]/m_L[1], mQS_L[2]/m_L[2], mQS_L[3]/m_L[3] ])
gamNR_S = 0.07 * np.array([mQS_S[0]/m_S[0], mQS_S[1]/m_S[1], mQS_S[2]/m_S[2], mQS_S[3]/m_S[3] ])


def DL(i): #allows me to grab the dipole centers and directions for each rod
    rod_cent_i = rod_centers[i : i+1, :]
    dipcent_i = dip_centers[i : i+1, :]
    vecs = L_vecs[i : i+1, :]
    return np.column_stack(( rod_cent_i, dipcent_i, vecs ))

def DS(i): #allows me to grab the dipole centers and directions for each rod
    rod_cent_i = rod_centers[i : i+1, :]
    dipcent_i = dip_centers[i : i+1, :]
    vecs = S_vecs[i : i+1, :]
    return np.column_stack(( rod_cent_i, dipcent_i, vecs ))

def make_g(mode_i, mode_j,m,k): #mode 1,2 are four columns: [sph_cent_x, sph_cent_y, vec_x, vec_y] and four rows corresponding to the four spheres
    k = np.real(k)
    r_ij = mode_i[0,2:4]-mode_j[0,2:4]  #distance between the nth and mth dipole
    mag_rij = np.linalg.norm(r_ij)
    if mag_rij == 0: g=0
    else:
        nhat_ij = r_ij / mag_rij
        xi = mode_i[0, 4:6]  
        xj = mode_j[0, 4:6]   
        xi_hat = xi/np.linalg.norm(xi)
        xj_hat = xj/np.linalg.norm(xj)
        xi_dot_nn_dot_xj = np.dot(xi_hat, nhat_ij)*np.dot(nhat_ij, xj_hat)
        nearField = ( 3.*xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat) ) / mag_rij**3
        intermedField = 1j*k*(3*xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat)) / mag_rij**2 
        farField = k**2*(xi_dot_nn_dot_xj - np.dot(xi_hat,xj_hat)) / mag_rij
        g =  e**2 * hbar_eVs**2 * ( nearField - intermedField - farField ) * np.exp(1j*k*mag_rij) 
    return -g/(m)

def make_H(k):
    H = np.zeros( (int(numIndModes*numSph),int(numIndModes*numSph)),dtype=complex) 
    w_thisround = k*c/np.sqrt(eps_b)*hbar_eVs #eV

    for i in range(0, numSph): #handle the on diagonal terms 
        gam_L = gamNR_L[i] + (np.real(w_thisround))**2*(2.0*e**2)/(3.0*m_L[i]*c**3)/hbar_eVs
        gam_S = gamNR_S[i] + (np.real(w_thisround))**2*(2.0*e**2)/(3.0*m_S[i]*c**3)/hbar_eVs

        H[ int(numIndModes*i)   , int(numIndModes*i)   ] = w0_L[i]**2 - 1j*gam_L*w_thisround
        H[ int(numIndModes*i+1) , int(numIndModes*i+1) ] = w0_S[i]**2 - 1j*gam_S*w_thisround

    for rod_i in range(0 , numSph-1): 
        for rod_j in range(1, numSph): 
            if rod_i < rod_j:
                ### L_i coupled with (L_1, S_1, ... L_N, S_N) ### 
                H[ int(numIndModes*rod_i), int(numIndModes*rod_j)  ] = make_g(mode_i=DL(i=rod_i), mode_j=DL(i=rod_j), m=m_L[rod_i], k=k)
                H[ int(numIndModes*rod_i), int(numIndModes*rod_j+1) ] = make_g(mode_i=DL(i=rod_i), mode_j=DS(i=rod_j), m=m_L[rod_i], k=k)
                
                ### S_i coupled with (L_1, S_1, ... L_N, S_N) ###
                H[ int(numIndModes*rod_i+1), int(numIndModes*rod_j)  ] = make_g(mode_i=DS(i=rod_i), mode_j=DL(i=rod_j), m=m_S[rod_i], k=k)
                H[ int(numIndModes*rod_i+1), int(numIndModes*rod_j+1)] = make_g(mode_i=DS(i=rod_i), mode_j=DS(i=rod_j), m=m_S[rod_i], k=k)

                ########## Now the opposite of the above terms ##########
                H[ int(numIndModes*rod_j)  , int(numIndModes*rod_i) ] = make_g(mode_i=DL(i=rod_j), mode_j=DL(i=rod_i), m=m_L[rod_j], k=k)
                H[ int(numIndModes*rod_j+1), int(numIndModes*rod_i) ] = make_g(mode_i=DS(i=rod_j), mode_j=DL(i=rod_i), m=m_S[rod_j], k=k)
                
                H[ int(numIndModes*rod_j)  , int(numIndModes*rod_i+1) ] = make_g(mode_i=DL(i=rod_j), mode_j=DS(i=rod_i), m=m_L[rod_j], k=k)
                H[ int(numIndModes*rod_j+1), int(numIndModes*rod_i+1) ] = make_g(mode_i=DS(i=rod_j), mode_j=DS(i=rod_i), m=m_S[rod_j], k=k)

        eigval, eigvec = np.linalg.eig(H)
    return eigval, eigvec, H

def interate():
    final_eigvals = np.zeros(np.int(numIndModes*numSph),dtype=complex)
    final_eigvecs = np.zeros( (np.int(numIndModes*numSph), np.int(numIndModes*numSph)), dtype=complex) 
    each_H = np.zeros( (np.int(numIndModes*numSph), np.int(numIndModes*numSph), np.int(numIndModes*numSph)), dtype=complex) 
    w_Lstart = -1j*gamNR_L[0]/2. + np.sqrt(-gamNR_L[0]**2/4.+w0_L[0]**2)
    w_Sstart = -1j*gamNR_S[0]/2. + np.sqrt(-gamNR_S[0]**2/4.+w0_S[0]**2)
    for mode in range(0,np.int(numIndModes*numSph)): #converge each mode individually         
        if mode == 0 or mode == 2 or mode == 4 or mode == 6: 
            eigval_hist = np.array([w_Lstart, w_Lstart*1.1],dtype=complex) 
        if mode == 1 or mode == 3 or mode == 5 or mode == 7:
            eigval_hist = np.array([w_Sstart, w_Sstart*1.1],dtype=complex) 
        eigvec_hist = np.zeros((int(numIndModes*numSph), 2))
        eigvec_hist[:,0] = 0.5
        vec_prec = np.zeros((int(numIndModes*numSph), 1))+10**(-prec)
        count = 0
        inercount = 1
        while np.abs((np.real(eigval_hist[0]) - np.real(eigval_hist[1])))  > 10**(-prec) and np.sum(np.abs((eigvec_hist[:,0] - eigvec_hist[:,1]))) > 10**(-prec):
            w_thisround = eigval_hist[0]
            if count > 100: 
               denom = ( eigval_hist[2] - eigval_hist[1] ) - ( eigval_hist[1] - eigval_hist[0] )
               w_thisround = eigval_hist[2] - ( eigval_hist[2] - eigval_hist[1] )**2 / denom 
            k = w_thisround/hbar_eVs*np.sqrt(eps_b)/c
            val, vec, H = make_H(k=k)
            amp = np.sqrt(np.abs(val))
            phi = np.arctan2(np.imag(val), np.real(val))
            energy = amp*np.cos(phi/2)
            post_sort_val = energy[energy.argsort()]
            post_sort_vec = vec[:,energy.argsort()]
            this_val = post_sort_val[mode]
            this_vec = post_sort_vec[:,mode]
            new_eigvals = this_val
            eigval_hist = np.append(new_eigvals, eigval_hist)
            eigvec_hist = np.column_stack((this_vec, eigvec_hist))
            each_H[mode,:,:] = H
            count = count + 1 
        final_eigvals[mode] = eigval_hist[0]
        final_eigvecs[:,mode] = eigvec_hist[:,0]
    return final_eigvals, final_eigvecs, each_H


def writeModes(mode):
    final_eigvals, final_eigvecs, H = interate()
    dip_ycoords = dip_centers[:,0]
    dip_zcoords = dip_centers[:,1]  
    w = np.real(final_eigvals[mode])
    v = np.real(final_eigvecs[:,mode])
    file = open(str('output_files/normal_mode_')+str(mode)+str('.txt'),'w')
    file.write( 'Sph. Center [nm]' + '\t' + '\t' + 'Vector Displacement [nm]' + '\n')
    for rod_i in range(0,numSph):
        DL_i = L_vecs[rod_i : rod_i+1, :]
        DS_i = S_vecs[rod_i : rod_i+1, :]
        mag_mode = (v[int(numIndModes*rod_i)]*DL_i + v[int(numIndModes*rod_i+1)]*DS_i)
        print(dip_ycoords[rod_i : rod_i+1], dip_zcoords[rod_i : rod_i+1], mag_mode[:,0], mag_mode[:,1])
        file.write("%.5e" % dip_ycoords[rod_i : rod_i+1] + '\t' + 
                   "%.5e" % dip_zcoords[rod_i : rod_i+1] + '\t' + 
                   "%.5e" % mag_mode[:,0] + '\t' + "%.5e" % mag_mode[:,1] + '\n')
    file.close()
#writeModes(mode=2)

def seeVectors(mode):
    final_eigvals, final_eigvecs, H = interate()
    L_vecs = inputs[:,4:6]
    S_vecs = inputs[:,6:8] 
    w = np.real(final_eigvals[mode])
    v = np.real(final_eigvecs[:,mode])

    dip_ycoords = dip_centers[:,0]
    dip_zcoords = dip_centers[:,1]  
   
    plt.subplot(1,numIndModes*numSph,mode+1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.title('%.2f eV' % (w), fontsize=18)
    plt.scatter(dip_ycoords, dip_zcoords,c='blue',s=50)

    for rod_i in range(0, numSph):
        DL_i = L_vecs[rod_i : rod_i+1, :]
        DS_i = S_vecs[rod_i : rod_i+1, :]
        mag_mode = (v[int(numIndModes*rod_i)]*DL_i + v[int(numIndModes*rod_i+1)]*DS_i)

        ymin = min(dip_ycoords)-1E-5; ymax = max(dip_ycoords)+1E-5
        zmin = min(dip_zcoords)-1E-5; zmax = max(dip_zcoords)+1E-5
        plt.quiver(dip_ycoords[rod_i : rod_i+1], dip_zcoords[rod_i : rod_i+1], mag_mode[:,0], mag_mode[:,1], pivot='mid', 
            width=.5, #shaft width in arrow units 
            scale=1., 
            headlength=5,
            headwidth=5.,#5.8
            minshaft=4., #4.1
            minlength=.1)
    plt.xlim([ymin, ymax])
    plt.ylim([zmin, zmax])
    plt.yticks([])
    plt.xticks([])
    #plt.show()
    return w, mag_mode

# fig = plt.figure(num=None, figsize=(12, 2), dpi=80, facecolor='w', edgecolor='k')   
# for mode in range(0,int(numIndModes*numSph)):
#     seeVectors(mode=mode)
# plt.show()


def find_supermasses():
    final_eigvals, final_eigvecs, H = interate()
    one_on_isolated_masses = np.array([1/m_L[0], 1/m_S[0], 1/m_L[1], 1/m_S[1], 1/m_L[2], 1/m_S[2], 1/m_L[3], 1/m_S[3]])
    R = np.real(final_eigvecs)
    Lam = np.matmul(np.linalg.inv(R), np.matmul(H[0,:,:], R))
    print(Lam)
    one_on_super_masses = np.matmul(np.linalg.inv(R), one_on_isolated_masses)
    print(1/one_on_isolated_masses)

def seeFields(mode):
    final_eigvals, final_eigvecs, H = interate()
    L_vecs = inputs[:,4:6]
    S_vecs = inputs[:,6:8] 
    w = np.real(final_eigvals[mode])
    v = np.real(final_eigvecs[:,mode])

    sph_xcoords = 0*dip_centers[:,0]
    sph_ycoords = dip_centers[:,0]
    sph_zcoords = dip_centers[:,1]  
    sphere_origins = np.column_stack((sph_xcoords, sph_ycoords, sph_zcoords))
    p = np.zeros((int(numSph), 3))
    for sph_i in range(0, int(numSph)):
        L_i = L_vecs[(sph_i) : (sph_i+1), :]
        S_i = S_vecs[(sph_i) : (sph_i+1), :]
        p[int(sph_i) : int(sph_i+1), 1:3] = ( v[int(numIndModes*sph_i)]*L_i + v[int(numIndModes*sph_i+1)]*S_i )

    ymin = min(sphere_origins[:,1])-1E-5; ymax = max(sphere_origins[:,1])+1E-5
    zmin = min(sphere_origins[:,2])-1E-5; zmax = max(sphere_origins[:,2])+1E-5

    x = 60e-07; 
    numPoints = 51
    y = np.linspace(ymin, ymax, numPoints ); z = np.linspace(zmin, zmax, numPoints )

    ### Efield for every dipole, [ which dipole, which y point, which z point ] ###
    Ex_field = np.zeros( (int(numSph), int(numPoints), int(numPoints)),dtype=complex)
    Ey_field = np.zeros( (int(numSph), int(numPoints), int(numPoints)),dtype=complex)
    Ez_field = np.zeros( (int(numSph), int(numPoints), int(numPoints)),dtype=complex)
    
    for which_dipole in range(0, int(numSph)):
        for which_y in range(0, int(numPoints)):
            for which_z in range(0, int(numPoints)):
                xval = x
                yval = y[which_y]
                zval = z[which_z]
                k = w/hbar_eVs/c
                point = np.array([xval, yval, zval])
                r = point - sphere_origins
                nhat = r/np.linalg.norm(r)
                nhat_dot_p = np.sum(nhat*p,axis=1)[:,np.newaxis]
                magr = np.linalg.norm(r,axis=1)[:,np.newaxis]
                nearField = ( 3*nhat * nhat_dot_p - p ) / magr**3
        #         intermedField1 = 1j*k*(3*rhat1 - np.dot(rhat1,p1)) / (np.linalg.norm(r1))**2
        #         farField1 = k**2*(rhat1 - np.dot(rhat1,p1)) / (np.linalg.norm(r1))
                Ex_field[which_dipole, which_z, which_y] = nearField[which_dipole,0]#(nearField1 + intermedField1 + farField1 ) * np.exp(1j*k*np.linalg.norm(r1))
                Ey_field[which_dipole, which_z, which_y] = nearField[which_dipole,1]#(nearField1 + intermedField1 + farField1 ) * np.exp(1j*k*np.linalg.norm(r1))
                Ez_field[which_dipole, which_z, which_y] = nearField[which_dipole,2]#(nearField1 + intermedField1 + farField1 ) * np.exp(1j*k*np.linalg.norm(r1))

    Extot = np.real(Ex_field[0,:,:]+Ex_field[1,:,:]+Ex_field[2,:,:]+Ex_field[3,:,:])
    #Eytot = np.real(Ey_field[whichsphere,:,:])#+Ey_field[1,:,:]+Ey_field[2,:,:]+Ey_field[3,:,:]+Ey_field[4,:,:]+Ey_field[5,:,:]+Ey_field[6,:,:]+Ey_field[7,:,:])
    #Eztot = np.real(Ez_field[whichsphere,:,:])

    plt.subplot(1,numIndModes*numSph,mode+1)
    ax = plt.gca()
    plt.title('%.2f eV' % (w), fontsize=18)
    plt.yticks([]); plt.xticks([])

    plt.imshow(Extot,
        alpha=0.4, 
        cmap='seismic',
        origin='lower',
        extent=[ymin,ymax,zmin,zmax]
        )

    pointsy = np.array([-90, 0, 90, 0, -90])*1E-7
    pointsz = np.array([0, 160, 0, -160, 0])*1E-7

    plt.plot(pointsy, pointsz, 'k:', linewidth=2.)
    plt.quiver(sphere_origins[:,1], sphere_origins[:,2], p[:,1], p[:,2], pivot='mid', 
        width=0.5, #shaft width in arrow units 
        scale=1, 
        headlength=5,
        headwidth=5.,
        minshaft=4., 
        minlength=.1)
    plt.xlim([ymin, ymax])
    plt.ylim([zmin, zmax])

# fig = plt.figure(num=None, figsize=(12, 3), dpi=80, facecolor='w', edgecolor='k')   
# for mode in range(0,int(numIndModes*numSph)):
#     seeFields(mode=mode)
# plt.show()
