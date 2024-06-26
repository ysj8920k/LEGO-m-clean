import numpy as np


def Type_finder(p1,p2,p3):
    l1 = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    l2 = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
    lengths=[l1,l2]
    
    #print("The max length")
    #print(np.max(lengths))
    #print("The min length")
    #print(np.min(lengths))
    
    ratio=np.max(lengths)/np.min(lengths)
    
    #print("The ratio")
    #print(ratio)
    
    if 0.75 <ratio< 1.25:
        Brick_type=1
    elif 1.75 <ratio< 2.25:
        Brick_type=2
    else:
        Brick_type=0
        #print('Brick_type error')
        
    return Brick_type

def get_slope(p1,p2,p3,type):
    #will always return the angle of the longest side or the minimum angle 

    l1 = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    l2 = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
    lengths=[l1,l2]

    if type==1:
        if p3[0]-p2[0] != 0:
            slope = (p3[1]-p2[1])/(p3[0]-p2[0])
        else:
            slope=0
    elif type==2:
        if lengths[0] < lengths[1] and p2[0]-p1[0] != 0:
            slope = (p2[1]-p1[1])/(p2[0]-p1[0])
        elif lengths[1] < lengths[0] and p3[0]-p2[0] != 0:
            slope = (p3[1]-p2[1])/(p3[0]-p2[0])
        else:
            slope=0




    return np.arctan(slope)*(180/np.pi)

def calc_length(x,y):
    l1 = np.sqrt((x[1] - x[0])**2 + (y[0] - y[1])**2)
    return l1