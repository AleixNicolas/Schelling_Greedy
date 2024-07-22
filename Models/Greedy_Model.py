import numpy as np
from numba import njit
import csv

#convolution product with non-periodic conditions compatible with numba
@njit
def convolve2d_numba(image, kernel_neighbors):
    m, n = image.shape
    output = np.zeros_like(image)
    for i in range(m):
        for j in range(n):
            sum_val = 0.0
            for ki in range(kernel_neighbors.shape[0]):
                for kj in range(kernel_neighbors.shape[1]):
                    ii = i + ki - 1
                    jj = j + kj - 1
                    if ii >= 0 and ii < m and jj >= 0 and jj < n:
                        sum_val += image[ii, jj] * kernel_neighbors[ki, kj]
            output[i, j] = sum_val
    return output

#Compute segregation as the fraction of same type of neighbors
@njit
def Compute_Segregation(p_positions, n_positions, p_neigh, n_neigh):
    return np.sum((p_positions*p_neigh+n_positions*n_neigh))/np.sum(((p_positions+n_positions)*(p_neigh+n_neigh)))

#if denominator=0 returns 0, noral division otherwise
@njit
def custom_divide(num,den):
    sol=num.copy()
    for i in range(len(num[:,0])):
        for j in range(len(num[0,:])):
            if den[i,j]==0:
                sol[i,j]=1
            else:
                sol[i,j]=num[i,j]/den[i,j]
    return sol

#Compute Global Happiness as the fraction of happy agents
@njit
def Compute_Happiness(position_matrix,kernel_neighbors):
    occupied_matrix= np.abs(position_matrix)
    happiness_matrix = position_matrix*convolve2d_numba(position_matrix,kernel_neighbors)
    happiness_matrix = happiness_matrix>=0
    happiness_matrix = happiness_matrix*occupied_matrix
    happiness = np.sum(happiness_matrix)/np.sum(occupied_matrix)
    return happiness_matrix, happiness

@njit
def Compute_Simulation(batch_size, kernel_neighbors, positions, p_positions, n_positions,indexp_o,indexp_e,r_index_i,p_r,G,S,H_matrix,H,alpha):
    #Initialize variables
    results = []
    step=0
    G_aux=np.zeros(len(indexp_e[:,0]))
    
    for i in range(batch_size):
        x_i = indexp_o[r_index_i[i], :].copy()
        # Determine if random agent is unhappy
        if H_matrix[x_i[0],x_i[1]]==0:
            available_positions=False
            
            #Determine the type of agent
            sign=positions[x_i[0],x_i[1]]
            
            #Initialize and update final positions final position when removing agent
            positions_f = positions.copy()
            positions_f[x_i[0],x_i[1]]=0
            p_positions_f = p_positions.copy()
            p_positions_f[x_i[0],x_i[1]]=0
            n_positions_f = n_positions.copy()
            n_positions_f[x_i[0],x_i[1]]=0
            
            #Compute G for all possible movements            
            for j, x_aux in enumerate(indexp_e):
                positions_aux=positions_f.copy()
                positions_aux[x_aux[0],x_aux[1]]=sign
                H_matrix_aux, H_aux=Compute_Happiness(positions_aux,kernel_neighbors)
                if H_matrix_aux[x_aux[0],x_aux[1]]==1:
                    p_positions_aux = p_positions_f.copy()
                    n_positions_aux = n_positions_f.copy()
                    if sign==1:
                        p_positions_aux[x_aux[0],x_aux[1]]=1
                    if sign==-1:
                        n_positions_aux[x_aux[0],x_aux[1]]=1
                    p_neigh_aux=convolve2d_numba(p_positions_aux, kernel_neighbors)
                    n_neigh_aux=convolve2d_numba(n_positions_aux, kernel_neighbors)
                    S_aux = Compute_Segregation(p_positions_aux, n_positions_aux, p_neigh_aux, n_neigh_aux)
                    G_aux[j]=-alpha*S_aux+(1-alpha)*H_aux
                    available_positions=True
                else:
                    G_aux[j]=-1000000
            
            #Select final position maximizing G
            if available_positions==True:
                step+=1
                threshold=np.max(G_aux)
                G_aux[G_aux<threshold]=0
                G_sum=np.sum(G_aux)
                G_aux=G_aux/G_sum
                p=0
                
                #If more than one possiblity randomly select one
                for j in range(len(G_aux)):
                    p+=G_aux[j]
                    if p>p_r[i]:
                        #Update state of the system and compute H and S
                        x_f=indexp_e[j,:]
                        positions=positions_f.copy()
                        positions[x_f[0],x_f[1]]=sign
                        H_matrix, H=Compute_Happiness(positions,kernel_neighbors)
                        p_positions = p_positions_f.copy()
                        n_positions = n_positions_f.copy()
                        if sign==1:
                            p_positions[x_f[0],x_f[1]]=1
                        if sign==-1:
                            n_positions[x_f[0],x_f[1]]=1
                        p_neigh=convolve2d_numba(p_positions, kernel_neighbors)
                        n_neigh=convolve2d_numba(n_positions, kernel_neighbors)
                        S = Compute_Segregation(p_positions, n_positions, p_neigh, n_neigh)
                        indexp_o[r_index_i[i], :] = x_f
                        indexp_e[j, :] = x_i
                        
                        results.append([i,step,H,S])

                        #Finish simulation if final state is reached                        
                        if H==1:
                            return results
                        
                        break
    return results

# Function to call with parameters and initial conditions
def Main(path,positions,kernel_neighbors,alpha,batch_size):
    #path: where to save the data, string
    #positions: initial positions of the different types of neighbors, 2D arrays with 0,1,-1
    #kernel neighbors: 2d array which determines which elements are sites are considered neighbors, 0,1
    #alpha: parameter of minimization function G, float from 0 to 1
    #batch_size: max number of attempted movements
    
    positions=np.copy(positions)
    G=len(positions[:,0])
    n_occupied=np.sum(np.abs(positions))
    
    #Generate matrixes with initail occupied and empty positions
    indexp_o = np.array([(i, j) for i, row in enumerate(positions) for j, val in enumerate(row) if val == 1 or val ==-1])
    indexp_e = np.array([(i, j) for i, row in enumerate(positions) for j, val in enumerate(row) if val == 0])
    
    #Generate matrixes with initial positions of positive and negative neighbors
    p_positions=np.where(positions == 1, 1, 0)
    n_positions=np.where(positions == -1, 1, 0)

    #Generate matrixes o number of initial positive and negative neighobrs
    p_neigh = convolve2d_numba(p_positions, kernel_neighbors)
    n_neigh = convolve2d_numba(n_positions, kernel_neighbors)
    
    #Compute inital segregation and happiness
    S=Compute_Segregation(p_neigh,n_neigh,p_positions,n_positions)
    H_matrix, H = Compute_Happiness(positions,kernel_neighbors)
    
    #Generate random values to choose which agent moves and where does it move to.
    r_index_i=np.random.randint(0,n_occupied,(batch_size))
    p_r=np.random.rand(batch_size)
    
    #Open csv file where the simulation values are to be saved
    with open(path,'w',newline='') as file:
        writer=csv.writer(file)
        
        writer.writerow([0,0,H,S])
        
        #Cumpute simulation
        results= Compute_Simulation(batch_size, kernel_neighbors, positions, p_positions, n_positions,indexp_o,indexp_e,r_index_i,p_r,G,S,H_matrix,H,alpha)
        
        for result in results:
            writer.writerow(result)

