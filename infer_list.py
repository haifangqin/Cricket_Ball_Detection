
import os
from mmdet.apis import init_detector, inference_detector,show_result_pyplot
import json
import glob

CLASSES = ('ball')


def infer(root_path, save_dir):
    # config_file = './configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
    config_file = './configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
    checkpoint_file = './tutorial_exps_refine/latest.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
    # model.CLASSES = 1
    # print(model)

    img_list = sorted(glob.glob(os.path.join(root_path, '*.png')))

    res_data = {}
    for img_path in img_list:
        result = inference_detector(model, img_path)
        # Let's plot the result
        # show_result_pyplot(model, img, result, score_thr=0.0)
        res_data[img_path.split('/')[-1]] = []
        if len(result) > 0:
            for cls, res in enumerate(result):
                if len(res) > 0:
                    for item in res:
                        res_data[img_path.split('/')[-1]].append(
                            {'bbox': list(item[0:4]), 'prob': item[4], 'cls': CLASSES[cls]})
    json_object = json.dumps(eval(str(res_data)), indent=4)
    # Writing to sample.json
    with open("{}.json".format(os.path.join(save_dir, root_path.split('/')[-1])), "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    vid_list = sorted(glob.glob(os.path.join('../datasets/select_shot_Frames/*')))
    vid_list = sorted(glob.glob(os.path.join('../datasets/INDvsSL_23-Output_15000_01/*')))
    save_dir = 'INDvsSL_23-Output_15000_01'
    for vid_path in vid_list:
        print(vid_path)
        infer(vid_path, save_dir)
        


