import time
import matplotlib.pyplot as plt
import numpy as np
n=10
lc=2.8553245961666107E-10 #lattice constant https://link-springer-com.proxy.lib.uwaterloo.ca/content/pdf/10.1007/s11467-011-0193-0
def ec2(ec1):
    return ec1*(1/(3**0.5/2))**(-6)
e_cor_Fe=4.122435278300438*1.6021773E-19
e_cor_Cr=3.8335*1.6021773E-19
ec_Fe_Fe=-0.78207*1.6021773E-19
ec_Cr_Cr=-0.32994*1.6021773E-19
e_ba=0.127682366510271*1.6021773E-19

def ec_bv_2(x):
    return (-0.02216+0.022382*x-0.01258*x**2)*1.6021773E-19 
def Hmix(x):git pull --rebase origin branchname
## EAB由三次拟合给出
    return (9.11 + 82.2*x - 145.12*(1 - 2*x)**2 + 7.92*(1 - 2*x)**3 + 44.91*(1 - 2*x)**4 + 33.18*(1 - 2*x)**5)*1.6021773E-19
def ec_AB(x):
    return (-0.73530+0.05305*x-0.029828*x**2)*1.6021773E-19

van_formation_Fe=2.55586*1.6021773E-19
ec_Fe_V=-0.22203*1.6021773E-19 ##文献不符合
def H(x):
    return (59.11 
            + 82.2 * x 
            - 145.12 * (1 - 2 * x)**2 
            + 7.92 * (1 - 2 * x)**3 
            + 44.91 * (1 - 2 * x)**4 
            + 33.18 * (1 - 2 * x)**5)
x=np.linspace(0.1, 0.4,20)
yy=((H(x)+(1-x)*e_cor_Fe+x*e_cor_Cr)/(x*(1-x))+4*ec_Fe_Fe+4*ec_Cr_Cr+ec2(3*ec_Fe_Fe+3*ec_Cr_Cr))/(8+(1/(3**0.5/2))**(-6)*6)
c=np.polyfit(x, yy, 2)
yc=c[0]*x**2+c[1]*x+c[2]

xx_test=np.linspace(0.1, 0.4,100)
ec_bv_t=e_ba-ec_Fe_Fe+ec_Fe_V+ec_AB(xx_test)
coeff_test=np.polyfit(xx_test,ec_bv_t , 2)
def ec_bv_1(x):    
    return (-0.10396+0.05305*x-0.02982*x**2)*1.6021773E-19

from numba import cuda
@cuda.jit(device=True)
def is_same_atom_pos_(i, j, k, ind, n, aa):
    kvec = 1
    jvec = 1
    ivec = 1

    if ind != 4:
        if ind in [0, 1, 2, 3]:
            kvec = 1
        else:
            kvec = -1

        if ind in [0, 2, 5, 7]:
            ivec = -1
        else:
            ivec = 1

        if ind in [0, 1, 5, 6]:
            jvec = -1
        else:
            jvec = 1

        # Modify the passed-in array 'aa' in place (no np.array)
        aa[0, 0], aa[0, 1], aa[0, 2], aa[0, 3] = i % n, j % n, k % n, ind
        aa[1, 0], aa[1, 1], aa[1, 2], aa[1, 3] = (i + ivec) % n, j % n, k % n, (ind - ivec) % n
        aa[2, 0], aa[2, 1], aa[2, 2], aa[2, 3] = i % n, (j + jvec) % n, k % n, (ind - jvec * 2) % n
        aa[3, 0], aa[3, 1], aa[3, 2], aa[3, 3] = (i + ivec) % n, (j + jvec) % n, k % n, (ind - jvec * 2 - ivec) % n
        aa[4, 0], aa[4, 1], aa[4, 2], aa[4, 3] = i % n, j % n, (k + kvec) % n, (ind + kvec * 5) % n
        aa[5, 0], aa[5, 1], aa[5, 2], aa[5, 3] = (i + ivec) % n, j % n, (k + kvec) % n, (ind - ivec + kvec * 5) % n
        aa[6, 0], aa[6, 1], aa[6, 2], aa[6, 3] = i % n, (j + jvec) % n, (k + kvec) % n, (ind - jvec * 2 + kvec * 5) % n
        aa[7, 0], aa[7, 1], aa[7, 2], aa[7, 3] = (i + ivec) % n, (j + jvec) % n, (k + kvec) % n, (ind - jvec * 2 - ivec + kvec * 5) % n
    else:
        # If ind == 4, return only one set of values
        aa[0, 0], aa[0, 1], aa[0, 2], aa[0, 3] = i, j, k, ind


@cuda.jit(device=True)
def pos_cuda(la, i, j, k, ind, nn, n, pos_la_U):
    if nn == 1:
        if ind == 4:
            # Fixed array for ind == 4 case
            pos_la_U[0, 0], pos_la_U[0, 1], pos_la_U[0, 2], pos_la_U[0, 3] = i, j, k, 0
            pos_la_U[1, 0], pos_la_U[1, 1], pos_la_U[1, 2], pos_la_U[1, 3] = i, j, k, 1
            pos_la_U[2, 0], pos_la_U[2, 1], pos_la_U[2, 2], pos_la_U[2, 3] = i, j, k, 2
            pos_la_U[3, 0], pos_la_U[3, 1], pos_la_U[3, 2], pos_la_U[3, 3] = i, j, k, 3
            pos_la_U[4, 0], pos_la_U[4, 1], pos_la_U[4, 2], pos_la_U[4, 3] = i, j, k, 5
            pos_la_U[5, 0], pos_la_U[5, 1], pos_la_U[5, 2], pos_la_U[5, 3] = i, j, k, 6
            pos_la_U[6, 0], pos_la_U[6, 1], pos_la_U[6, 2], pos_la_U[6, 3] = i, j, k, 7
            pos_la_U[7, 0], pos_la_U[7, 1], pos_la_U[7, 2], pos_la_U[7, 3] = i, j, k, 8
        else:
            kvec = 1 if ind in [0, 1, 2, 3] else -1
            ivec = -1 if ind in [0, 2, 5, 7] else 1
            jvec = -1 if ind in [0, 1, 5, 6] else 1

            pos_la_U[0, 0], pos_la_U[0, 1], pos_la_U[0, 2], pos_la_U[0, 3] = i % n, j % n, k % n, 4
            pos_la_U[1, 0], pos_la_U[1, 1], pos_la_U[1, 2], pos_la_U[1, 3] = (i + ivec) % n, j % n, k % n, 4
            pos_la_U[2, 0], pos_la_U[2, 1], pos_la_U[2, 2], pos_la_U[2, 3] = i % n, (j + jvec) % n, k % n, 4
            pos_la_U[3, 0], pos_la_U[3, 1], pos_la_U[3, 2], pos_la_U[3, 3] = (i + ivec) % n, (j + jvec) % n, k % n, 4
            pos_la_U[4, 0], pos_la_U[4, 1], pos_la_U[4, 2], pos_la_U[4, 3] = i % n, j % n, (k + kvec) % n, 4
            pos_la_U[5, 0], pos_la_U[5, 1], pos_la_U[5, 2], pos_la_U[5, 3] = (i + ivec) % n, j % n, (k + kvec) % n, 4
            pos_la_U[6, 0], pos_la_U[6, 1], pos_la_U[6, 2], pos_la_U[6, 3] = i % n, (j + jvec) % n, (k + kvec) % n, 4
            pos_la_U[7, 0], pos_la_U[7, 1], pos_la_U[7, 2], pos_la_U[7, 3] = (i + ivec) % n, (j + jvec) % n, (k + kvec) % n, 4
    else:
        # Case where nn != 1
        pos_la_U[0, 0], pos_la_U[0, 1], pos_la_U[0, 2], pos_la_U[0, 3] = (i + 1) % n, j % n, k % n, ind
        pos_la_U[1, 0], pos_la_U[1, 1], pos_la_U[1, 2], pos_la_U[1, 3] = (i - 1) % n, j % n, k % n, ind
        pos_la_U[2, 0], pos_la_U[2, 1], pos_la_U[2, 2], pos_la_U[2, 3] = i % n, (j - 1) % n, k % n, ind
        pos_la_U[3, 0], pos_la_U[3, 1], pos_la_U[3, 2], pos_la_U[3, 3] = i % n, (j + 1) % n, k % n, ind
        pos_la_U[4, 0], pos_la_U[4, 1], pos_la_U[4, 2], pos_la_U[4, 3] = i % n, j % n, (k - 1) % n, ind
        pos_la_U[5, 0], pos_la_U[5, 1], pos_la_U[5, 2], pos_la_U[5, 3] = i % n, j % n, (k + 1) % n, ind


def is_same_atom_pos(i,j,k,ind):
    kvec=1
    jvec=1
    ivec=1
    if ind!=4:
        if ind in [0,1,2,3]:
            kvec=1
        else:
            kvec=-1
        if ind in [0,2,5,7]:
            ivec=-1
        else:
            ivec=1
        if ind in [0,1,5,6]:
            jvec=-1
        else:
            jvec=1
        aa=np.array([[i,j,k,ind],[i+ivec,j,k,ind-ivec],[i,j+jvec,k,ind-jvec*2],[i+ivec,j+jvec,k,ind-jvec*2-ivec],[i,j,k+kvec,ind+kvec*5],[i+ivec,j,k+kvec,ind-ivec+kvec*5],[i,j+jvec,k+kvec,ind-jvec*2+kvec*5],[i+ivec,j+jvec,k+kvec,ind-jvec*2-ivec+kvec*5]])  
        for iiii in range(8):
            for jjjj in range(3):
                aa[iiii][jjjj]=aa[iiii][jjjj]%n
                
        return aa
    else:
       return np.array([[i,j,k,ind]])
def is_same_atom(i,j,k,ind,ii,jj,kk,iind):
    kvec=1
    jvec=1
    ivec=1
    if ind!=4:
        if ind in [0,1,2,3]:
            kvec=1
        else:
            kvec=-1
        if ind in [0,2,5,7]:
            ivec=-1
        else:
            ivec=1
        if ind in [0,1,5,6]:
            jvec=-1
        else:
            jvec=1
        aa=np.array([[i,j,k,ind],[i+ivec,j,k,ind-ivec],[i,j+jvec,k,ind-jvec*2],[i+ivec,j+jvec,k,ind-jvec*2-ivec],[i,j,k+kvec,ind+kvec*5],[i+ivec,j,k+kvec,ind-ivec+kvec*5],[i,j+jvec,k+kvec,ind-jvec*2+kvec*5],[i+ivec,j+jvec,k+kvec,ind-jvec*2-ivec+kvec*5]])  
        for iiii in range(8):
            for jjjj in range(3):
                aa[iiii][jjjj]=aa[iiii][jjjj]%n
        ac=False      
        for i in range(8):
            if ii == aa[i][0] and jj == aa[i][1] and kk == aa[i][2] and iind == aa[i][3]:
                ac=True
        return ac
    else:
        if i==ii and j==jj and k==kk and ind==iind:
            return True
        else:
            return False    
def atom_parallel(la,i,j,k,ind):
    kvec=1
    jvec=1
    ivec=1
    if ind!=4:
        if ind in [0,1,2,3]:
            kvec=1
        else:
            kvec=-1
        if ind in [0,2,5,7]:
            ivec=-1
        else:
            ivec=1
        if ind in [0,1,5,6]:
            jvec=-1
        else:
            jvec=1
        aa=np.array([[i,j,k,ind],[i+ivec,j,k,ind-ivec],[i,j+jvec,k,ind-jvec*2],[i+ivec,j+jvec,k,ind-jvec*2-ivec],[i,j,k+kvec,ind+kvec*5],[i+ivec,j,k+kvec,ind-ivec+kvec*5],[i,j+jvec,k+kvec,ind-jvec*2+kvec*5],[i+ivec,j+jvec,k+kvec,ind-jvec*2-ivec+kvec*5]])  
        for iiii in range(8):
            for jjjj in range(3):
                aa[iiii][jjjj]=aa[iiii][jjjj]%n
        for i1 in range(7):
            la[aa[i1+1][0]][aa[i1+1][1]][aa[i1+1][2]][aa[i1+1][3]]=la[i][j][k][ind]
        return 
    else:
        return
def pos_(la,i,j,k,ind,nn):
    if nn==1:
        if ind==4:
            return np.array([[i,j,k,0],[i,j,k,1],[i,j,k,2],[i,j,k,3],[i,j,k,5],[i,j,k,6],[i,j,k,7],[i,j,k,8]])
        else:
            kvec=1
            jvec=1
            ivec=1

            if ind in [0,1,2,3]:
                kvec=1
            else:
                kvec=-1
            if ind in [0,2,5,7]:
                ivec=-1
            else:
                ivec=1
            if ind in [0,1,5,6]:
                jvec=-1
            else:
                jvec=1
            aa=np.array([[i,j,k,4],[i+ivec,j,k,4],[i,j+jvec,k,4],[i+ivec,j+jvec,k,4],[i,j,k+kvec,4],[i+ivec,j,k+kvec,4],[i,j+jvec,k+kvec,4],[i+ivec,j+jvec,k+kvec,4]])  
            for iiii in range(8):
                for jjjj in range(3):
                    aa[iiii][jjjj]=aa[iiii][jjjj]%n
            return aa
    else:
        aa=np.array([[i+1,j,k,ind],[i-1,j,k,ind],[i,j-1,k,ind],[i,j+1,k,ind],[i,j,k-1,ind],[i,j,k+1,ind]])
        for iiii in range(6):
            for jjjj in range(3):
                aa[iiii][jjjj]=aa[iiii][jjjj]%n
        return aa       



from numba import cuda
import numpy as np

# Define the kernel

@cuda.jit
def energy_parallel_kernel(la, lat, ucoef_partial, ucoef_temp_partial, jcoef_partial, jcoef_temp_partial, n):
    idx = cuda.grid(1)
    total_threads = cuda.gridsize(1)
    total_elements = n * n * n

    for index in range(idx, total_elements, total_threads):
        # Compute itp, jtp, ktp from the flattened index
        itp = index // (n * n)
        rem = index % (n * n)
        jtp = rem // n
        ktp = rem % n

        indtp = 4  # Per your code

        # Get pos_la_U using the device function
        pos_la_U = cuda.local.array((8, 4), dtype=numba.int32)
        
        # Compute neighbor indices based on indtp
        if indtp == 4:
            for ttt in range(8):
                if ttt < 4:
                    pos_la_U[ttt, 0] = itp
                    pos_la_U[ttt, 1] = jtp
                    pos_la_U[ttt, 2] = ktp
                    pos_la_U[ttt, 3] = ttt
                else:
                    pos_la_U[ttt, 0] = itp
                    pos_la_U[ttt, 1] = jtp
                    pos_la_U[ttt, 2] = ktp
                    pos_la_U[ttt, 3] = ttt + 1
        else:
            kvec = 1 if indtp in [0, 1, 2, 3] else -1
            ivec = -1 if indtp in [0, 2, 5, 7] else 1
            jvec = -1 if indtp in [0, 1, 5, 6] else 1
            pos_la_U[0, 0] = itp
            pos_la_U[0, 1] = jtp
            pos_la_U[0, 2] = ktp
            pos_la_U[0, 3] = 4

            pos_la_U[1, 0] = (itp + ivec) % n
            pos_la_U[1, 1] = jtp
            pos_la_U[1, 2] = ktp
            pos_la_U[1, 3] = 4

            pos_la_U[2, 0] = itp
            pos_la_U[2, 1] = (jtp + jvec) % n
            pos_la_U[2, 2] = ktp
            pos_la_U[2, 3] = 4

            pos_la_U[3, 0] = (itp + ivec) % n
            pos_la_U[3, 1] = (jtp + jvec) % n
            pos_la_U[3, 2] = ktp
            pos_la_U[3, 3] = 4

            pos_la_U[4, 0] = itp
            pos_la_U[4, 1] = jtp
            pos_la_U[4, 2] = (ktp + kvec) % n
            pos_la_U[4, 3] = 4

            pos_la_U[5, 0] = (itp + ivec) % n
            pos_la_U[5, 1] = jtp
            pos_la_U[5, 2] = (ktp + kvec) % n
            pos_la_U[5, 3] = 4

            pos_la_U[6, 0] = itp
            pos_la_U[6, 1] = (jtp + jvec) % n
            pos_la_U[6, 2] = (ktp + kvec) % n
            pos_la_U[6, 3] = 4

            pos_la_U[7, 0] = (itp + ivec) % n
            pos_la_U[7, 1] = (jtp + jvec) % n
            pos_la_U[7, 2] = (ktp + kvec) % n
            pos_la_U[7, 3] = 4

        # Initialize per-thread partial sums
        ucoef_sum = 0.0
        ucoef_temp_sum = 0.0
        jcoef_sum = 0.0
        jcoef_temp_sum = 0.0

        # Perform computations for 8 neighbors
        for i in range(8):
            la_idx0 = pos_la_U[i, 0]
            la_idx1 = pos_la_U[i, 1]
            la_idx2 = pos_la_U[i, 2]
            la_idx3 = pos_la_U[i, 3]

            la_current = la[itp, jtp, ktp, indtp]
            la_neighbor = la[la_idx0, la_idx1, la_idx2, la_idx3]
            lat_current = lat[itp, jtp, ktp, indtp]
            lat_neighbor = lat[la_idx0, la_idx1, la_idx2, la_idx3]

            ucoef_sum += la_current**2 * la_neighbor + la_current * la_neighbor**2
            ucoef_temp_sum += lat_current**2 * lat_neighbor + lat_current * lat_neighbor**2
            jcoef_sum += la_current * la_neighbor
            jcoef_temp_sum += lat_current * lat_neighbor

        # Store partial sums in output arrays
        ucoef_partial[index] = ucoef_sum
        ucoef_temp_partial[index] = ucoef_temp_sum
        jcoef_partial[index] = jcoef_sum
        jcoef_temp_partial[index] = jcoef_temp_sum

import numba
def energy(la,lat,i,j,k,ind,x,x_temp):
    
    K_1nn=1/4*(ec_Fe_Fe+ec_Cr_Cr+2*ec_AB(x))  -(ec_Fe_V+ec_bv_1(x))
    K_temp_1nn=1/4*(ec_Fe_Fe+ec_Cr_Cr+2*ec_AB(x_temp)) -(ec_Fe_V+ec_bv_1(x_temp))
    U_1nn=1/4*(ec_Fe_Fe-ec_Cr_Cr)-1/2*(ec_Fe_V-ec_bv_1(x))
    U_temp_1nn=1/4*(ec_Fe_Fe-ec_Cr_Cr)-1/2*(ec_Fe_V-ec_bv_1(x_temp))
    J_1nn=1/4*(ec_Fe_Fe+ec_Cr_Cr-2*ec_AB(x))
    J_temp_1nn=1/4*(ec_Fe_Fe+ec_Cr_Cr-2*ec_AB(x_temp))
    
    K_2nn=1/4*(ec2(ec_Fe_Fe+ec_Cr_Cr+2*ec_AB(x)))  -(ec2(ec_Fe_V)+ec_bv_2(x))
    K_temp_2nn=1/4*(ec2(ec_Fe_Fe+ec_Cr_Cr+2*ec_AB(x_temp)))  -(ec2(ec_Fe_V)+ec_bv_2(x_temp))
    U_2nn=1/4*ec2((ec_Fe_Fe-ec_Cr_Cr))-1/2*(ec2(ec_Fe_V)-ec_bv_2(x))
    U_temp_2nn=1/4*ec2((ec_Fe_Fe-ec_Cr_Cr))-1/2*(ec2(ec_Fe_V)-ec_bv_2(x_temp))
    J_2nn=1/4*ec2((ec_Fe_Fe+ec_Cr_Cr-2*ec_AB(x)))
    J_temp_2nn=1/4*ec2((ec_Fe_Fe+ec_Cr_Cr-2*ec_AB(x_temp)))
    dH=0
    
    dH+=(K_temp_1nn- K_1nn)*8*(n*n*n-1)
    
    #### 计算U U必须要计算所有晶格
    #### Run the CUDA kernel to calculate U and J contributions across the lattice
    la_device = cuda.to_device(la)
    lat_device = cuda.to_device(lat)
    total_elements = n * n * n
    
    # Create device arrays for partial sums
    ucoef_partial = cuda.device_array(total_elements, dtype=np.float32)
    ucoef_temp_partial = cuda.device_array(total_elements, dtype=np.float32)
    jcoef_partial = cuda.device_array(total_elements, dtype=np.float32)
    jcoef_temp_partial = cuda.device_array(total_elements, dtype=np.float32)

    # Launch CUDA kernel
    threads_per_block = 512
    blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block
    energy_parallel_kernel[blocks_per_grid, threads_per_block](
        la_device, lat_device, ucoef_partial, ucoef_temp_partial, jcoef_partial, jcoef_temp_partial, n
    )
    
    # Copy results back to host
    ucoef_partial_host = ucoef_partial.copy_to_host()
    ucoef_temp_partial_host = ucoef_temp_partial.copy_to_host()
    jcoef_partial_host = jcoef_partial.copy_to_host()
    jcoef_temp_partial_host = jcoef_temp_partial.copy_to_host()
    
    # Sum the partial results
    ucoef = np.sum(ucoef_partial_host)
    ucoef_temp = np.sum(ucoef_temp_partial_host)
    jcoef = np.sum(jcoef_partial_host)
    jcoef_temp = np.sum(jcoef_temp_partial_host)
    dH+=U_temp_1nn*ucoef_temp-U_1nn*ucoef+J_temp_1nn*jcoef_temp-J_1nn*jcoef       
   
    return dH
def energy_vacancy(la,i,j,k,ind,ii,jj,kk,iind,x):
    dH=0    
    U_1nn=1/4*(ec_Fe_Fe-ec_Cr_Cr)-1/2*(ec_Fe_V-ec_bv_1(x))
    J_1nn=1/4*(ec_Fe_Fe+ec_Cr_Cr-2*ec_AB(x))
    U_2nn=1/4*ec2((ec_Fe_Fe-ec_Cr_Cr))-1/2*(ec2(ec_Fe_V)-ec_bv_2(x))
    J_2nn=1/4*ec2((ec_Fe_Fe+ec_Cr_Cr-2*ec_AB(x)))
    latp=np.copy(la)   
    latp[i][j][k][ind]=latp[ii][jj][kk][iind]
    latp[ii][jj][kk][iind]=0
    ucoef=0
    jcoef=0
    pos_va=pos_(la, i, j, k, ind, 1)
    pos_atom=pos_(la, ii, jj, kk, ind, 1)
    for itp in range(8):
       ucoef+=latp[i][j][k][ind]**2*latp[pos_va[itp][0]][pos_va[itp][1]][pos_va[itp][2]][pos_va[itp][3]]+ latp[i][j][k][ind]*latp[pos_va[itp][0]][pos_va[itp][1]][pos_va[itp][2]][pos_va[itp][3]]**2  
       ucoef-=la[i][j][k][ind]**2*la[pos_va[itp][0]][pos_va[itp][1]][pos_va[itp][2]][pos_va[itp][3]]+ la[i][j][k][ind]*la[pos_va[itp][0]][pos_va[itp][1]][pos_va[itp][2]][pos_va[itp][3]]**2
       ucoef+=latp[ii][jj][kk][iind]**2*latp[pos_atom[itp][0]][pos_atom[itp][1]][pos_atom[itp][2]][pos_atom[itp][3]]+ latp[ii][jj][kk][iind]*latp[pos_atom[itp][0]][pos_atom[itp][1]][pos_atom[itp][2]][pos_atom[itp][3]]**2  
       ucoef-=la[ii][jj][kk][iind]**2*la[pos_atom[itp][0]][pos_atom[itp][1]][pos_atom[itp][2]][pos_atom[itp][3]]+ la[ii][jj][kk][iind]*la[pos_atom[itp][0]][pos_atom[itp][1]][pos_atom[itp][2]][pos_atom[itp][3]]**2  
       jcoef+=latp[i][j][k][ind]*latp[pos_va[itp][0]][pos_va[itp][1]][pos_va[itp][2]][pos_va[itp][3]]
       jcoef-=la[i][j][k][ind]*la[pos_va[itp][0]][pos_va[itp][1]][pos_va[itp][2]][pos_va[itp][3]]
       jcoef+=latp[ii][jj][kk][iind]*latp[pos_atom[itp][0]][pos_atom[itp][1]][pos_atom[itp][2]][pos_atom[itp][3]]
       jcoef-=la[ii][jj][kk][iind]*la[pos_atom[itp][0]][pos_atom[itp][1]][pos_atom[itp][2]][pos_atom[itp][3]]
    dH+=U_1nn*ucoef+J_1nn*jcoef
    
    
    return dH
##概率再分配
def reprob():
    aa=np.random.rand()*16
    if aa<8:
        return 4
    elif aa<9 :
        return 0
    elif aa<10 :
        return 1
    elif aa<11 :
        return 2
    elif aa<12 :
        return 3
    elif aa<13 :
        return 5
    elif aa<14 :
        return 6
    elif aa<15 :
        return 7  
    elif aa<16 :
        return 8
##metropolis


@cuda.jit
def yita_(la, la_t, n, x, yita_device):
    i, j, k = cuda.grid(3)  # 获取3D线程索引

    # 确保线程在数组的合法范围内
    if i < n and j < n and k < n:
        for ind in range(9):
            # 创建一个局部数组用于存储 is_same_atom_pos_ 的结果
            if ind!=4:
                ac = cuda.local.array((8, 4), dtype=numba.int32)
                is_same_atom_pos_(i, j, k, ind,n, ac)  # 假设这个函数是 CUDA 兼容的设备函数
                cc=8
            else:
                ac = cuda.local.array((1, 4), dtype=numba.int32)
                is_same_atom_pos_(i, j, k, ind,n, ac)  # 假设这个函数是 CUDA 兼容的设备函数
                cc=1
            is_same = False
            a_num = 0
           
            # 检查原子位置是否已经标记
            for tp in range(cc):
                if la_t[ac[tp,0], ac[tp,1], ac[tp,2], ac[tp,3]] != 0:
                    is_same = True
                    break
                else:
                    la_t[ac[tp,0], ac[tp,1], ac[tp,2], ac[tp,3]] = 1

            if not is_same:
                if la[i, j, k, ind] == -1:
                    # 使用局部数组存储 pos_cuda 的结果
                    pp = cuda.local.array((8, 4), dtype=numba.int32)
                    pos_cuda(la, i, j, k, ind, 1,10, pp)  # 假设这个函数是 CUDA 兼容的设备函数

                    for ttp in range(8):
                        if la[pp[ttp,0], pp[ttp,1], pp[ttp,2], pp[ttp,3]] == 1:
                            a_num += 1
                    
                    # 使用原子操作更新 yita_device 以避免竞态条件
                    contribution = 1 / int((x * n * n * n * 2)) * (1 - a_num / 8 / (1-x))
                    cuda.atomic.add(yita_device, 0, contribution)






def metropl(T,dt1,dt):
    la=np.zeros((n,n,n,n))
    lat=np.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ind in range(9):  
                    la[i][j][k][ind]=1#Fe 1 index=4 centre atom
    
    vi,vj,vk,vind=int(np.random.rand()*n),int(np.random.rand()*n),int(np.random.rand()*n),int(np.random.rand()*9)
    la[vi][vj][vk][vind]=0 #vacancy
    atom_parallel(la, vi, vj, vk, vind)
    num_Cr=0
    for temp in range(num_Cr):
        is_Cr=False
        while not is_Cr:
            i,j,k,ind=int(np.random.rand()*n),int(np.random.rand()*n),int(np.random.rand()*n),int(np.random.rand()*9)
            if la[i][j][k][ind]==1:
                la[i][j][k][ind]=-1
                atom_parallel(la, i, j, k, ind)
                is_Cr=True
        #np.unravel_index(abs(la).argmin(), la.shape)
    NA=n*n*n*2-num_Cr #Fe
    NB=num_Cr#Cr
    x=NB/(n*n*n*2)
    
    kb=1.346E-23
    num_filp=10000
    dmiu=1.5*1.6021773E-19###chemical potential 
    cdmiu=0.1*1.6021773E-19
    xmiu=[]
    dddmiu=[]
    yitay=[]
    x_total=[]
    yita_total=[]
    for dmiui in range(dt1,dt):
        tc=time.time()
        dmiu=0.03*1.6021773E-19*dmiui
        dddmiu.append(dmiu)
        x_temp=0 
        x_temp1=[]
        yita_temp=[]
        for temp in range(5000):
            i,j,k,ind=int(np.random.rand()*n),int(np.random.rand()*n),int(np.random.rand()*n),reprob()
            x_temp=x
        
            if la[i][j][k][ind]!=0:
                if la[i][j][k][ind]==1:
                    x_temp =x+1/(n*n*n*2)
                    dmiuco=1
                else:
                    x_temp=x-1/(n*n*n*2)
                    dmiuco=-1
                lat=np.copy(la)
                lat[i][j][k][ind]=lat[i][j][k][ind]*(-1)
                ddH=energy(la,lat,i,j,k,ind,x,x_temp)
                if ddH+2*dmiu*dmiuco<=0 or np.random.rand()<=np.exp(-(ddH+2*dmiu*dmiuco)/kb/T):
                    la[i][j][k][ind]*=-1
                    atom_parallel(la, i, j, k, ind)
                    x=x_temp
                
            else:        
                iss_atom=True
                while iss_atom:
                    ii,jj,kk,iind=int(np.random.rand()*n),int(np.random.rand()*n),int(np.random.rand()*n),reprob()
                    iss_atom=is_same_atom(i, j, k, ind, ii, jj, kk, iind)
                ddH=energy_vacancy(la,i,j,k,ind,ii,jj,kk,iind,x)
                if ddH<=0 or np.random.rand()<=np.exp(-(ddH)/kb/T):
                    la[i][j][k][ind]=la[ii][jj][kk][iind]
                    la[ii][jj][kk][iind]=0
                    atom_parallel(la, i, j, k, ind)
                    atom_parallel(la, ii, jj, kk, iind)
        
        la_t = np.zeros((n, n, n, n), dtype=np.int32) 
        la_device = cuda.to_device(la)
        la_t_device = cuda.to_device(la_t)
                
        yita_host = np.array([0.0], dtype=np.float32)
        yita_device = cuda.to_device(yita_host)
        threads_per_block = (8, 8, 8)
        blocks_per_grid = (n // threads_per_block[0] + 1, n // threads_per_block[1] + 1, n // threads_per_block[2] + 1)
        yita_[blocks_per_grid, threads_per_block](la_device, la_t_device, n, x, yita_device)
                
        yita_host = yita_device.copy_to_host()
        yita_temp.append(yita_host)
   
        x_total.append(x)
        yita_total.append(yita_temp)
                
        xmiu.append(x)
        la_t=np.zeros((n,n,n,n))
        
  
        #print(x,dmiu,yita)  
        print(time.time()-tc)                   
    return [xmiu,dddmiu,np.array(x_total),np.array(yita_total)]

c=[]
d=[]
T=np.linspace(450,700,80)

for i in range(len(T)):
     temp=metropl(T[i],1,25)
     c.append(temp[2])
     d.append(temp[3])