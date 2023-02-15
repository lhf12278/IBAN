import cv2
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def Filter_Fudiao(src_img):
    # filter=np.array([[-1,0,0],[0,0,0],[0,0,1]])
    filter = np.array([[-1, 0], [0, 1]])
    row=src_img.shape[0]
    col=src_img.shape[1]
    new_img=np.zeros([row,col],dtype=np.uint8)
    for i in range(row-1):
        for j in range(col-1):
            new_value = np.sum(src_img[i:i + 2, j:j + 2] * filter) + 128  # point multiply
            if new_value > 255:
                new_value = 255
            elif new_value < 0:
                new_value = 0
            else:
                pass
            new_img[i, j]=new_value
    return new_img

root_path = '/data/dataset/video_cross-modal/VCM-HIT'
for i in range(597, 600, 1):
    no1_dir = str(i).rjust(4, '0')
    no1_dirpath = 'F:/VCM-HIT/' + no1_dir

    isExists = os.path.exists(no1_dirpath)
    if not isExists:#文件夹不存在
        continue;

    # dirpath_pick = os.listdir(dirpath)
    # dirpath_pick.sort(key=lambda x: int(x))
    no2_dir_rgb_path = os.path.join(no1_dirpath, 'rgb')
    no2_dir_ir_path  = os.path.join(no1_dirpath, 'ir')

######################################################################################################################

    no3_dir_rgb = os.listdir(no2_dir_rgb_path)
    for j_r in no3_dir_rgb:
        create_path = os.path.join(root_path, no1_dir, 'rgb', j_r)#新文件夹路径
        isExists = os.path.exists(create_path)
        if not isExists:
            os.makedirs(create_path)#创建文件夹

        no3_dir_rgb_path = os.path.join(no2_dir_rgb_path, j_r)#原文件夹路径
        imgpath_pick = os.listdir(no3_dir_rgb_path)
        for k in imgpath_pick:
            readpath = os.path.join(no3_dir_rgb_path, k)#图片读取路径
            src_img = cv2.imread(readpath)
            gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            new_img = Filter_Fudiao(gray_img)
            cv2.waitKey()

            save_path=os.path.join(create_path, k)#图片存储路径
            cv2.imwrite(save_path, new_img)
            print(save_path)

#####################################################################################################################

    no3_dir_ir = os.listdir(no2_dir_ir_path)
    for j_r in no3_dir_ir:
        create_path = os.path.join(root_path, no1_dir, 'ir', j_r)#文件夹路径
        isExists = os.path.exists(create_path)
        if not isExists:
            os.makedirs(create_path)#创建文件夹

        no3_dir_ir_path = os.path.join(no2_dir_ir_path, j_r)
        imgpath_pick = os.listdir(no3_dir_ir_path)
        for k in imgpath_pick:
            readpath = os.path.join(no3_dir_ir_path, k)#图片读取路径
            src_img = cv2.imread(readpath)
            gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            new_img = Filter_Fudiao(gray_img)
            cv2.waitKey()

            save_path=os.path.join(create_path, k)#图片存储路径
            cv2.imwrite(save_path, new_img)
            print(save_path)




