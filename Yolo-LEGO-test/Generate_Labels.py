import cv2
import os
import numpy as np

def gen_img_mask(path,savepath,y_res,x_res):
    print(path)

    mask_cords = np.loadtxt(path, comments="#", delimiter=" ", unpack=False)

    #print(mask_cords)
    cords=[]
    
    for i in range(len(mask_cords)):
        if mask_cords[i,0]!=0:
            mask_cords_list=[mask_cords[i,0]-1,mask_cords[i,8],mask_cords[i,9],mask_cords[i,5],mask_cords[i,6],
                        mask_cords[i,11],mask_cords[i,12],mask_cords[i,14],mask_cords[i,15]]
            #mask_cords_list=[mask_cords[i,0]-1,mask_cords[i,5],mask_cords[i,6],mask_cords[i,8],mask_cords[i,9],
            #            mask_cords[i,11],mask_cords[i,12],mask_cords[i,14],mask_cords[i,15]]
            #mask_cords_list=np.array([[mask_cords[i,8],mask_cords[i,9]],[mask_cords[i,5],mask_cords[i,6]],
            #            [mask_cords[i,11],mask_cords[i,12]],[mask_cords[i,14],mask_cords[i,15]]])

            cords.append(mask_cords_list)
         
    print(cords)
    with open(savepath, 'w') as f:
        for line in cords:
            f.write(("".join(str(line)) + "\n").replace("]","").replace("[","").replace(",","")) 
        f.close()

            
    print(savepath)

#y_resolution=512
#x_resolutiion=512

y_resolution=512
x_resolutiion=512

dir="Yolo-LEGO-test/data/labels"
val="/val/"
train="/train/"
test="/test/"

save_dir="Yolo-LEGO-test/data/segmentation_labels"

dir_val=dir+val
dir_train=dir+train

val_dir_list=os.listdir(dir_val)
train_dir_list=os.listdir(dir_train)

#for i in range(0,1):
for i in range(0,len(train_dir_list)):
    gen_img_mask(dir_train+train_dir_list[i],save_dir+train+train_dir_list[i],y_resolution,x_resolutiion)

for i in range(0,len(val_dir_list)):
    gen_img_mask(dir_val+val_dir_list[i],save_dir+val+val_dir_list[i],y_resolution,x_resolutiion)




#if __name__ == 'main': Skal lige fikses


