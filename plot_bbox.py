import json
import cv2
import os
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import glob

def plot_bbox(img_dir, bbox_path, prob):
    save_dir = os.path.join(img_dir + '_{}'.format(prob))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(bbox_path, 'r') as f:
        bbox = json.load(f)
    # print(bbox.keys())
    for key in bbox.keys():
        count = 0
        img = cv2.imread(os.path.join(img_dir, key))
        for bbox_item in bbox[key]:
            if bbox_item['prob'] > prob:
                cv2.rectangle(img, (int(bbox_item['bbox'][0]), int(bbox_item['bbox'][1])),
                              (int(bbox_item['bbox'][2]), int(bbox_item['bbox'][3])), color=(0, 0, 255), thickness=2)
                text = str(bbox_item['prob'])[:4]
                cv2.putText(img, text, (int(bbox_item['bbox'][0]), int(bbox_item['bbox'][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)

                print(os.path.join(save_dir, key.replace('.png', '.jpg')), text)
                # print(np.mean(img))
        cv2.imwrite(os.path.join(save_dir, key.replace('.png', '.jpg')), img)

        count += 1
        if count == 0:
        #     cmd = 'ffmpeg -i {} {}'.format(os.path.join(img_dir, key), os.path.join(save_dir, key.replace('.png', '.jpg')))
            cmd = "cp {} {}".format(os.path.join('INDvsSL_23-Output_0.8', key.replace('.png', '.jpg')), os.path.join(save_dir, key.replace('.png', '.jpg')))
            subprocess.call(cmd, shell=True)

def plot_bbox_max_pro(img_dir, bbox_path, save_dir):
    with open(bbox_path, 'r') as f:
        bbox = json.load(f)
    # print(bbox.keys())
    for key in bbox.keys():
        count = 0
        prob = 0.5
        best_item = None
        img = cv2.imread(os.path.join(img_dir, key))
        for bbox_item in bbox[key]:
            if bbox_item['prob'] > prob:
                prob = bbox_item['prob']
                best_item = bbox_item
                count += 1

        if count == 0:
            cmd = "cp {} {}".format(os.path.join(img_dir, key), os.path.join(save_dir, key.replace('.png', '.jpg')))
            print(cmd)
            subprocess.call(cmd, shell=True)
        else:
            text = str(best_item['prob'])[:4]
            cv2.putText(img, text, (int(best_item['bbox'][0]), int(best_item['bbox'][1])), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255), 1)
            cv2.rectangle(img, (int(best_item['bbox'][0]), int(best_item['bbox'][1])),
                          (int(best_item['bbox'][2]), int(best_item['bbox'][3])), color=(0, 0, 255), thickness=2)
            cv2.imwrite(os.path.join(save_dir, key.replace('.png', '.jpg')), img)

def DaVinci_infer():
    img_list = sorted(glob.glob("INDvsSL_23-Output_1000/*.png"))
    save_dir = 'INDvsSL_23-Output_1000_DaVinci'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img_path in img_list:
        cmd = 'DaVinci_ISR_General_20220622\DaVinci_ISR_General_Tool_X2.exe {} {}'.\
            format(img_path, img_path.replace('INDvsSL_23-Output_1000', 'INDvsSL_23-Output_1000_DaVinci'))
        print(cmd)
        subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    # DaVinci_infer()
    # img_dir = 'INDvsSL_23-Output'
    # bbox_path = 'INDvsSL_23-Output.json'
    # plot_bbox(img_dir, bbox_path, prob=0.5)


    #vid_list = sorted(glob.glob("../datasets/select_shot_Frames/*"))
    #for img_dir in vid_list:
        #print(img_dir)
    img_dir = '/home/ubuntu/HFR/datasets/3rdODIOutputRecord_600_120_imgs_for_det'
    bbox_path = '3rdODIOutputRecord_600_120_imgs_for_det_val_v5_v2.json'
    save_dir = '3rdODIOutputRecord_600_120_imgs_for_det_val_v5_v2'
    prob = 0.5
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #plot_bbox(img_dir, bbox_path, prob)
    plot_bbox_max_pro(img_dir, bbox_path, save_dir)



    # img = cv2.imread(os.path.join(img_dir, '00000130.png'))
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()


