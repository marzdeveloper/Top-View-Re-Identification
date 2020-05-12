import os, glob
from PIL import Image
import numpy as np
import cv2
from matplotlib import cm

dizio={}

def preprocess(base_path, data_path):
    basename=os.path.basename(data_path)
    out_path=base_path+"preprocessed/"+basename+"/"

    if os.path.exists(out_path):
        print("skip")
        return

    os.makedirs(out_path,exist_ok=True)

    folders=glob.glob(data_path+"/*")
    print(basename,"-",len(folders),"detections:")
    dizio[basename]=len(folders)

    for ind,f in enumerate(folders):
        base_folder=os.path.basename((f))
        print(ind,"Folder ",f)
        out_folder=out_path+base_folder+"/"


        os.makedirs(out_folder, exist_ok=True)
        imgs = glob.glob(f+"/*_rgb.png")     #_depth.png
        print(" - founded {} images".format(len(imgs)))
        for i in imgs:
            d=i[:-7]+"depth.png"
            img=Image.open(i)
            depth=Image.open(d)
            img_np_or=np.array(img)
            depth_np_or = np.array(depth)

            depth_np = depth_np_or[20:220, 20:300]
            img_np = img_np_or[20:220, 20:300]



            mask_depth = (depth_np < 1600) * 255

            mask_depth2 = (depth_np < 2000) * 255

            contours, hierarchy = cv2.findContours(mask_depth.astype('uint8') , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Check if contours exist:
            if len(contours) != 0:
                # Search biggest area contour
                c = max(contours, key=cv2.contourArea)
                # Computer the area of the contour
                area = cv2.contourArea(c)

                #Creo maschera dal contorno
                mask_contour = np.zeros(mask_depth.shape)
                mask_contour = cv2.drawContours(mask_contour, [c], -1, (1), -1)
                mask_contour = mask_contour.astype("uint8")
                img_contour = np.zeros(img_np.shape)
                img_contour = cv2.drawContours(img_contour, [c], -1, (1,1,1), -1)
                img_contour = img_contour.astype("uint8")


                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                #print(cX,cY)    # 98 38 --> asse x - asse y
                #cv2.circle(img_np, (cX, cY), 7, (255, 255, 255), -1)


                ini=cX-100
                if ini<0:           ini=0
                if ini>(240-200):   ini=240-200
                fin = ini + 200

                img_np *= img_contour
                img_cropped_np=img_np[:,ini:fin,:]
                img_cropped = Image.fromarray(img_cropped_np)

                #JET

                #SOGLIA=np.max(depth_np)
                SOGLIA = 2800

                depth_npt1=depth_np
                depth_npt1[depth_npt1>SOGLIA]=SOGLIA
                depth_npt2=SOGLIA-depth_npt1
                depth_npt2 = depth_npt2 * 255 // np.max(depth_npt2)

                depth_npt2=depth_npt2.astype("uint8")
                #mask_contour2 = mask_contour.astype("uint8")
                depth_npt2 *= mask_contour

                #depth_npt2 = depth_npt2 * 256 // 1000
                jet1 = cm.jet(depth_npt2)
                jet2 = jet1 * 255
                jet3 = jet2[:, ini:fin, :]
                img_jet = Image.fromarray(np.uint8(jet3))

                img_jet = img_jet.convert('RGB')
                #img_jet.show()

                #SAVING
                img_jet.save(out_folder + os.path.basename(d))
                img_cropped.save(out_folder + os.path.basename(i))
                #img_jet.save("test/" + os.path.basename(d))


def person_detection(mask, rgb):

    contours, hierarchy = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contours exist:
    if len(contours) != 0:
        # Search biggest area contour
        c = max(contours, key=cv2.contourArea)
        # Computer the area of the contour
        area = cv2.contourArea(c)

        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # print(cX,cY)    # 98 38 --> asse x - asse y

        x1, y1, w, h = cv2.boundingRect(c)
        # TODO Mancano le if di controllo su w ed h per capire se sono utili o meno
        # Find the end of the bounding box:
        x2 = x1 + w
        y2 = y1 + h

        # CROP
        if cX < 120:
            ini = 0
        else:
            ini = cX - 120
        fin = ini + 240
        #img_cropped = img_np[:, ini:fin, :]




base_path="E:/MassimoMartini/Re-id Rocco/Febbraio/"

folders=glob.glob(base_path+"*_clean")
for f in folders:
    preprocess(base_path, f)

print("Statistics:")
for k in dizio:
    print(k,"=",dizio[k],"detections")

print("Done!")
