import cv2
import os
import numpy as np

def gen_img_mask(path,savepath,y_res,x_res):
    print(path)
    img = np.zeros((x_res,y_res,3), dtype=np.uint8)
    mask_cords = np.loadtxt(path, comments="#", delimiter=" ", unpack=False)

    print(mask_cords)
    
    for i in range(len(mask_cords)):
        if mask_cords[i,0]!=0:
            mask_cords_list=np.array([[mask_cords[i,1],mask_cords[i,2]],[mask_cords[i,3],mask_cords[i,4]],
                        [mask_cords[i,5],mask_cords[i,6]],[mask_cords[i,7],mask_cords[i,8]]])
            #mask_cords_list=np.array([[mask_cords[i,5],mask_cords[i,6]],[mask_cords[i,8],mask_cords[i,9]],
            #            [mask_cords[i,11],mask_cords[i,12]],[mask_cords[i,14],mask_cords[i,15]]])
            #mask_cords_list=np.array([[mask_cords[i,8],mask_cords[i,9]],[mask_cords[i,5],mask_cords[i,6]],
            #            [mask_cords[i,11],mask_cords[i,12]],[mask_cords[i,14],mask_cords[i,15]]])

            
            mask_cords_list=mask_cords_list*x_res
            int_mask_cords_list = mask_cords_list.astype(int)
            #print("bf_sorting")
            #print(int_mask_cords_list)
            #int_mask_cords_list= int_mask_cords_list[int_mask_cords_list[:,0].argsort()]
            #int_mask_cords_list= int_mask_cords_list[int_mask_cords_list[:,1].argsort()]
            #print("after sorting")
            #print(int_mask_cords_list)


            cv2.fillPoly(img, pts=[int_mask_cords_list],color=(255,255,255))

    cv2.imshow('Webcam', img)
    cv2.waitKey()

    print(savepath)
    cv2.imwrite(savepath.replace('.txt','.png'), img) 

    

    




#y_resolution=512
#x_resolutiion=512

y_resolution=512
x_resolutiion=512

dir="Yolo-LEGO-test/data/segmentation_labels"
#dir="Yolo-LEGO-test/data/segmentation_labels"

val="/val/"
train="/train/"
test="/test/"

save_dir="Yolo-LEGO-test/mask"

dir_val=dir+val
dir_train=dir+train

val_dir_list=os.listdir(dir_val)
train_dir_list=os.listdir(dir_train)

for i in range(0,len(train_dir_list)):
    gen_img_mask(dir_train+train_dir_list[i],save_dir+train+train_dir_list[i],y_resolution,x_resolutiion)

for i in range(0,len(val_dir_list)):
    gen_img_mask(dir_val+val_dir_list[i],save_dir+val+val_dir_list[i],y_resolution,x_resolutiion)




#if __name__ == 'main': Skal lige fikses


