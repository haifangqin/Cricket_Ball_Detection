
import os
from mmdet.apis import init_detector, inference_detector,show_result_pyplot
import json
import glob

CLASSES = ('ball')

#root_path = '../datasets/cricket/INDvsSL_23-Output/'
#root_path = '../../4kLIVE/datasets/4K_HFR/3rdODIOutputRecord_14000/'
#root_path = '../datasets/select_shot_Frames/shot_55'
#root_path = '../../4kLIVE/datasets/4K_HFR/3rdODIOutputRecord_600_120'
#root_path = '../datasets/select_shot_Frames_Val_75/'
root_path = '/home/ubuntu/HFR/datasets/3rdODIOutputRecord_600_120_imgs_for_det'
list_file = '/home/ubuntu/HFR/datasets/3rdODIOutputRecord_600_120_imgs_for_det.txt'
config_file = './configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
checkpoint_file = './tutorial_exps_refine_v2/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
print(model)

with open(list_file, 'r') as fin:
    img_list = fin.readlines()


#img_list = sorted(glob.glob(os.path.join(root_path, '*.png')))

res_data = {}
for img_path in img_list:
    img_path = img_path.strip() + '.png'
    img_path = os.path.join(root_path, img_path)
    result = inference_detector(model, img_path)
    # Let's plot the result
    # show_result_pyplot(model, img, result, score_thr=0.0)
    res_data[img_path.split('/')[-1]] = []
    if len(result) > 0:
        for cls, res in enumerate(result):
            if len(res) > 0:
                for item in res:
                    res_data[img_path.split('/')[-1]].append({'bbox': list(item[0:4]), 'prob': item[4], 'cls': CLASSES[cls]})
json_object = json.dumps(eval(str(res_data)), indent=4)
# Writing to sample.json
#with open("3rdODIOutputRecord_600_120_v4.json", "w") as outfile:
#with open("3rdODIOutputRecord_14000_v4.json", "w") as outfile:
with open("3rdODIOutputRecord_600_120_imgs_for_det_val_v5_v2.json", "w") as outfile:
    outfile.write(json_object)


