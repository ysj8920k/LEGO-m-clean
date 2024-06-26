from robodk.robolink import *      # RoboDK's API
from robodk.robomath import *      # Math toolbox for robots
import numpy as np


# Start the RoboDK API:
RDK = Robolink()

# Get the robot items by name:
robot = RDK.Item('UR10', ITEM_TYPE_ROBOT)

home= RDK.Item('Home')

Pick_base= RDK.Item('Pick_Base')
Place_base= RDK.Item('Place_Base')
Mag_base=RDK.Item('Mag_base')
Test1 = RDK.Item('test1')
Test2 = RDK.Item('test2')

World=RDK.Item('World',ITEM_TYPE_FRAME)
Ref_Pick=RDK.Item('Pick_Mat',ITEM_TYPE_FRAME)
Ref_Place=RDK.Item('Build_plate',ITEM_TYPE_FRAME)
Ref_Mags=RDK.Item('Mags',ITEM_TYPE_FRAME)
Placed_bricks = RDK.Item('Placed_bricks', ITEM_TYPE_FOLDER)

def runmode():
    while True:
        input_var=input("To run LIVE press \"y\" or to run OFFLINE press \"n\": ")
        if input_var =="y":
            print("\n")
            print("ONLINE")
            return 1
        elif input_var=="n":
            print("\n")
            print("OFFLINE")
            return 0

        else:
            print("You have to choose yes or no by typing \"y\" or \"n\" \n")
        
        break
def tocontinue():
    while True:
        input_var=input("If pickup/place is correct press \"y\": ")
        if input_var =="y":
            print("\n")

        else:
            print("You have to choose yes or no by typing \"y\" or \"n\" \n")
        
        break

def pick_place(frame,x,y,z,a,b,c,speed):
    robot.setFrame(frame)
    robot.setSpeed(speed)
    pose_ref=robot.Pose()
    Pick=Mat(pose_ref)

    pos_pick=Pick.Pos()
    pos_pick=[x,y,z]
    Pick.setPos(pos_pick)
    
    Pick=Pick*rotx(a*pi/180)*roty(b*pi/180)*rotz(c*pi/180)

    robot.MoveL(Pick)

    robot.setFrame(RDK.Item('UR10 Base'))

def main_robot(runmode):

    #webcam = cv2.VideoCapture(1)

    if runmode==1:
        # Update connection parameters if required:
        # robot.setConnectionParams('192.168.2.35',30000,'/', 'anonymous','')

        # Connect to the robot using default IP
        success = robot.Connect()  # Try to connect once
        #success robot.ConnectSafe() # Try to connect multiple times
        status, status_msg = robot.ConnectedState()
        if status != ROBOTCOM_READY:
            # Stop if the connection did not succeed
            print(status_msg)
            raise Exception("Failed to connect: " + status_msg)

        # This will set to run the API programs on the robot and the simulator (online programming)
        RDK.setRunMode(RUNMODE_RUN_ROBOT)
        # Note: This is set automatically when we Connect() to the robot through the API


        joints_ref = robot.Joints()
        print("Ready to start")
        tocontinue()    
    else:
        RDK.setRunMode(RUNMODE_SIMULATE)
    #TEMP
    array_ins=np.genfromtxt('generated_instructions0.csv', delimiter=',')
    
    speed_normal=10
    speed_place=10
    Tool_length=300 #CHECK BEFORE RUNNING ### but run with the safety first
    
    robot.setSpeed(10)

    
    robot.MoveL(Test1)
    tocontinue()
    robot.MoveL(Test2)
    tocontinue()
    robot.setFrame(RDK.Item('UR10 Base'))
    robot.setSpeed(50)
    

if __name__ == '__main__':
    mode=runmode()
    main_robot(mode)#THIS CAN RUN LIVE
    

