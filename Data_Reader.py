# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 14:32:07 2024

@author: usuario
"""

import numpy as np
import pandas as pd
import csv
from numba import njit

kernel_neighbors = np.array([[1,1,1],[1,0,1],[1,1,1]])
boundary_conditions ="fill"

input_file_name="Experimental Data/match_sms_202312_12.csv"
output_file_name = "Experimental Data/Board_"

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

def Compute_team_neighbors(positions,kernel_neighbors,boundary_conditions):
    neighbors=np.abs(convolve2d_numba(positions,kernel_neighbors,boundary_conditions))
    return neighbors

#Compute Global Happiness as the fraction of happy agents
@njit
def Compute_Happiness(position_matrix,kernel_neighbors):
    occupied_matrix= np.abs(position_matrix)
    happiness_matrix = position_matrix*convolve2d_numba(position_matrix,kernel_neighbors)
    happiness_matrix = happiness_matrix>=0
    happiness_matrix = happiness_matrix*occupied_matrix
    happiness = np.sum(happiness_matrix)/np.sum(occupied_matrix)
    return happiness_matrix, happiness

#Compute segregation as the fraction of same type of neighbors
@njit
def Compute_Segregation(p_positions, n_positions, p_neigh, n_neigh):
    return np.sum((p_positions*p_neigh+n_positions*n_neigh))/np.sum(((p_positions+n_positions)*(p_neigh+n_neigh)))

def string_to_matrix(input_string):
    L=20    
    
    array_2d= np.array(eval(input_string))

    # Create a 20x20 array initialized with -1
    output_array = [[0]*L for _ in range(L)]

    # Mapping for values
    value_mapping = {'RH': -1, 'RS': -1, 'BH': 1, 'BS': 1}

    # Iterate over each element of array_2d
    for i in range(L):
        for j in range(L):
            # Get the value from array_2d
            value = array_2d[i, j]
            # Update the corresponding element in output_array based on value_mapping
            output_array[i][j] = value_mapping.get(value, 0)

    # Convert output_array to a NumPy array
    output_array = np.array(output_array)
    
    return output_array
    
def import_experimental_data(file_name):
    match_finale = (pd.read_csv(file_name)
                             .query("picture_happiness >0 ")
                             .query("picture_segregation >0 ")
                             .query("picture_happiness <= 1 ")
                             .query("picture_segregation <1 ")
            )
    
    match_finale["picture_upload_time"] = pd.to_datetime(match_finale["picture_upload_time"])
    match_finale = match_finale.sort_values("picture_upload_time")
    match_finale = match_finale.reset_index(drop=True)
    
    df=match_finale[['board_id','picture_table_status',"picture_upload_time"]]
    positions = []
    
    for board_id in df['board_id'].unique():
        filtered_df = df[df['board_id']==board_id]
        filtered_df = filtered_df.sort_values("picture_upload_time")
        list_aux=[]
        for index, row in filtered_df.iterrows():
            list_aux.append(string_to_matrix(row['picture_table_status']))
        positions.append(list_aux)
          
    return positions

def Steps_Computations(positions,k):
    current_position_matrix=np.array(positions[k-1,:,:])
    steps=0
    for i in range(len(positions[:k,0,0])-1):
        position_matrix=np.array(positions[i,:,:])
        steps=int(0.5*np.sum(np.absolute(current_position_matrix-position_matrix)))
        for j in range(3):
            position_matrix=np.rot90(position_matrix)
            steps=min(steps,int(0.5*np.sum(np.absolute(current_position_matrix-position_matrix))))
        if steps==0:
            break        
        
    return steps


    
def Main():
    positions = import_experimental_data(input_file_name)
    initial_position_matrix=np.zeros((len(positions),20,20))
    for i in range(len(positions)):
        with open(str(output_file_name)+str(i)+".csv",'w',newline='') as file:
            writer=csv.writer(file)
            initial_position_matrix[i,:,:]=np.array(positions[i][0])
            total_steps=0
            previous_position_matrix=np.copy(initial_position_matrix[i,:,:])
            
            #Generate matrixes with initial positions of positive and negative neighbors
            p_positions=np.where(previous_position_matrix == 1, 1, 0)
            n_positions=np.where(previous_position_matrix == -1, 1, 0)
        
            #Generate matrixes o number of initial positive and negative neighobrs
            p_neigh = convolve2d_numba(p_positions, kernel_neighbors)
            n_neigh = convolve2d_numba(n_positions, kernel_neighbors)
            
            happiness_matrix, happiness =Compute_Happiness(previous_position_matrix,kernel_neighbors)
            segregation=Compute_Segregation(p_positions, n_positions, p_neigh, n_neigh)
            writer.writerow([total_steps,happiness,segregation])
            k=1
            
            for element in positions[i]:
                position_matrix = np.copy(element)
                steps=Steps_Computations(np.array(positions[i]),k)
                k+=1
                total_steps+=steps
                if steps>0 and steps < 100:
                    #Generate matrixes with initial positions of positive and negative neighbors
                    p_positions=np.where(position_matrix == 1, 1, 0)
                    n_positions=np.where(position_matrix == -1, 1, 0)
                
                    #Generate matrixes o number of initial positive and negative neighobrs
                    p_neigh = convolve2d_numba(p_positions, kernel_neighbors)
                    n_neigh = convolve2d_numba(n_positions, kernel_neighbors)
                    
                    happiness_matrix, happiness =Compute_Happiness(position_matrix,kernel_neighbors)
                    segregation=Compute_Segregation(p_positions, n_positions, p_neigh, n_neigh)
                    writer.writerow([total_steps,happiness,segregation])
                previous_position_matrix=np.copy(position_matrix)
    
                
    return initial_position_matrix

Main()