import numpy as np
import os
import pandas as pd
import sys
import cProfile
import pstats
from Models import Greedy_Model
from Models import Schelling_Fast_Model

#Number of simulations for each initial conditions
iterations= 1000
#Max number of attempted movements
batch_size= 1000000

#Define which types of simulations to run
Greedy=True
Classic_Schelling=True

input_path="Experimental Data"
output_path="Simulations"

kernel_neighbors = np.array([[1,1,1],[1,0,1],[1,1,1]])
alpha_values=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]

#Determine initial positions
Random_Initial_Positions=False
rho=0.2
L=20
boards=[0,1,2,3]


def Generate_Random_Grid(rho,L):
    N=L*L
    A=np.ones(int(N*(1-rho)/2),dtype=int)
    B=-np.ones(int(N*(1-rho)/2),dtype=int)
    zeros=np.zeros(N-len(A)-len(B),dtype=int)
    aux=np.concatenate((A,B))
    aux=np.concatenate((aux,zeros))
    np.random.shuffle(aux)
    return np.reshape(aux,(L,L))

#Create initial positions matrix from position string
def string_to_matrix(input_string):
    L=20    
    array_2d= np.array(eval(input_string))
    output_array = [[0]*L for _ in range(L)]
    value_mapping = {'RH': -1, 'RS': -1, 'BH': 1, 'BS': 1}
    for i in range(L):
        for j in range(L):
            value = array_2d[i, j]
            output_array[i][j] = value_mapping.get(value, 0)
    output_array = np.array(output_array)
    
    return output_array

#Generate position string from file
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
        positions.append(string_to_matrix(filtered_df.iloc[0]['picture_table_status']))
    positions_array= np.array(positions)
    return positions_array

def Main():
    
    #Determine initial positions
    if Random_Initial_Positions==True:
        #Create directory to save data
        path=str(output_path)+"/Random_Initial_Conditions"
        if not os.path.exists(path):
            os.makedirs(str(path))
            
        for i in range(iterations):
            initial_positions_matrix=Generate_Random_Grid(rho,L)
            
            #Create directory for Classic Schelling simulations and compute corresponding simulations 
            if Classic_Schelling==True:
                path_1=str(path)+"/Classic_Schelling"
                if not os.path.exists(path_1):
                    os.makedirs(str(path_1))
                file_path=str(path_1)+"/"+str(i)+".csv"
                Schelling_Fast_Model.Main(file_path,initial_positions_matrix,kernel_neighbors,batch_size)
                
            #Create directory for greedy simulations and compute corresponding simulations                 
            if Greedy==True:
                path_1=str(path)+"/Greedy"
                for alpha in alpha_values:
                    path_2=str(path_1)+"/alpha_"+str(alpha)
                    if not os.path.exists(path_2):
                        os.makedirs(str(path_2))
                    sys.stdout.flush()
                    sys.stdout.write("\rboard: "+str(i) + " L_r_b " + " alpha="+str(alpha) +" "+str(i+1)+"/"+str(iterations))
                    file_path=str(path_2)+"/"+str(i)+".csv"
                    Greedy_Model.Main(file_path,initial_positions_matrix,kernel_neighbors,alpha,batch_size)

    else:
        initial_positions_matrix=import_experimental_data(str(input_path)+"/match_sms_202312_12.csv")

        #Compute simulations for all boards
        for board in boards:
            
            #Create directories to save data
            path=str(output_path)+"/board_"+str(board)
            if not os.path.exists(path):
                os.makedirs(str(path))
            
            #Create directory for schelling simulations and compute corresponding  simulations
            if Classic_Schelling==True:
                path_1=str(path)+"/Classic_Schelling"
                if not os.path.exists(path_1):
                    os.makedirs(str(path_1))
                for i in range(iterations):
                    print(board,i)
                    file_path=str(path_1)+"/"+str(i)+".csv"
                    Schelling_Fast_Model.Main(file_path,initial_positions_matrix[board,:,:],kernel_neighbors,batch_size)
            
            #Create directory for greedy simulations and compute corresponding simulations                 
            if Greedy==True:
                path_1=str(path)+"/Greedy"
                for alpha in alpha_values:
                    path_2=str(path_1)+"/alpha_"+str(alpha)
                    if not os.path.exists(path_2):
                        os.makedirs(str(path_2))
                    for i in range(iterations):
                        sys.stdout.flush()
                        sys.stdout.write("\rboard: "+str(board) + " L_r_b " + " alpha="+str(alpha) +" "+str(i+1)+"/"+str(iterations))
                        file_path=str(path_2)+"/"+str(i)+".csv"
                        Greedy_Model.Main(file_path,initial_positions_matrix[board,:,:],kernel_neighbors,alpha,batch_size)



with cProfile.Profile() as pr:
    Main()
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats(10)
stats.dump_stats(filename='stats.prof')
stats.dump_stats(filename='stats.prof')


