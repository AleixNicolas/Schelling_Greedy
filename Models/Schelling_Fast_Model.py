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
def Compute_Simulation(batch_size, kernel_neighbors, positions,p_positions, n_positions, p_neigh, n_neigh, happiness_matrix, happiness, indexp_o, indexp_e, r_index_i, r_index_f, S):
    #Initialize variables
    results = []
    step=0
    
    for i in range(batch_size):
        x_i = indexp_o[r_index_i[i], :].copy()
        # Determine if random agent is unhappy
        if happiness_matrix[x_i[0], x_i[1]]==0:
            x_f = indexp_e[r_index_f[i], :].copy()
            
            #Determine the type of agent
            sign = positions[x_i[0], x_i[1]]
            
            #Initialize and update final positions final position when removing agent
            positions_f=positions.copy()
            positions_f[x_i[0], x_i[1]] = 0
            positions_f[x_f[0], x_f[1]] = sign
            p_positions_f=p_positions.copy()
            n_positions_f=n_positions.copy()
            if sign ==+1:
                p_positions_f[x_i[0], x_i[1]] = 0
                p_positions_f[x_f[0], x_f[1]] = 1
            if sign ==-1:
                n_positions_f[x_i[0], x_i[1]] = 0
                n_positions_f[x_f[0], x_f[1]] = 1
            happiness_matrix_f, happiness_f = Compute_Happiness(positions_f,kernel_neighbors)
            
            #Update state of the system and compute H and S if new site is happy
            if happiness_matrix_f[x_f[0], x_f[1]]==1:
                step += 1
                positions=positions_f.copy()
                p_positions=p_positions_f.copy()
                n_positions=n_positions_f.copy()
                p_neigh=convolve2d_numba(p_positions, kernel_neighbors)
                n_neigh=convolve2d_numba(n_positions, kernel_neighbors)
                happiness_matrix=happiness_matrix_f.copy()
                happiness=happiness_f
                S = Compute_Segregation(p_positions, n_positions, p_neigh, n_neigh)
                indexp_o[r_index_i[i], :] = x_f
                indexp_e[r_index_f[i], :] = x_i
                results.append([i, step, happiness, S])
                
                #Finish simulation if final state is reached  
                if happiness==1:
                    return results, positions
    return results, positions

# Function to call with parameters and initial conditions
def Main(path,positions,kernel_neighbors, batch_size):
    #path: where to save the data, string
    #positions: initial positions of the different types of neighbors, 2D arrays with 0,1,-1
    #kernel neighbors: 2d array which determines which elements are sites are considered neighbors, 0,1
    #batch_size: max number of attempted movements
    
    positions=np.copy(positions)
    L=len(positions[:,0])
    n=L*L
    n_occupied=np.sum(np.abs(positions))
    n_empty=n-n_occupied
    
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
    happiness_matrix, happiness = Compute_Happiness(positions,kernel_neighbors)
    
    #Generate random values to choose which agent moves and where does it move to.
    r_index_f=np.random.randint(0,n_empty,(batch_size))
    r_index_i=np.random.randint(0,n_occupied,(batch_size))
    
    #Open csv file where the simulation values are to be saved
    with open(path,'w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow([0,0,happiness,S])
        
        #Cumpute simulation
        results, positions= Compute_Simulation(batch_size, kernel_neighbors, positions,p_positions, n_positions, p_neigh, n_neigh, happiness_matrix, happiness, indexp_o, indexp_e, r_index_i, r_index_f, S)
            
        if np.sum(np.abs(positions))!=320:
                print('error')
            
        for result in results:
            writer.writerow(result)
            