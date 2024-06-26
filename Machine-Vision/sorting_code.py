import MV_Main_belt as mv
import cv2


def belt_sorting(Mode=1,sorting_spaces=4,minframes=100):
    webcam = cv2.VideoCapture(1)
    webcam.set(cv2.CAP_PROP_EXPOSURE, -5)
    Magazines=[]
    for i in range(sorting_spaces):
        Magazines.append[[0]['Empty']]



    while True:
        results=mv.main_mv_belt(minframes,webcam)

        for rows in results:
            if all(rows['Standard Deviation x'],rows['Standard Deviation y'],rows['Standard Deviation Angle'])<1:
                
                if Mode==1: 
                    sort_cls=rows['Brick Colour']  
                elif Mode==2:
                    sort_cls=rows['Brick Type']
                elif Mode==3:
                    sort_cls=rows['Brick Type'] + " " + rows['Brick Colour']
                
                Mag_loc=Magazines[1].index(sort_cls)
                
                if Mag_loc is None:
                    Mag_loc=Magazines[1].index('Empty')

                print("Robot does something")### this where is does something 
                #If it is a success 
                Magazines[1,Mag_loc]=sort_cls
                Magazines[0,Mag_loc]+=1

                



if __name__ == '__main__':

    belt_sorting()