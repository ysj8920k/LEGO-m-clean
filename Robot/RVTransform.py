import numpy as np

def Transform(x,y,A):
    z=0
    vector=[x,y,z,1]
    T=np.array([[1,0,0,0],[0,1,0,0],[0, 0, 1, 0],[0, 0, 0, 1]])
    
    #T=np.array([[0,1,0,347.2999],[-1,0,0,-392.8348],[0, 0, 1, 145],[0, 0, 0, 1]])
    robot_coords=T.dot(vector)
    
    #A=-180-A ##CHECK THIS ONE
    return robot_coords,A