import os
import os.path as osp
import math
import argparse
from multiprocessing import Process, Queue
from tqdm import tqdm
import numpy as np
import megengine as mge
from megengine import jit
from config import config
from nms_wrapper import nms
from set_cpu_nms import set_cpu_nms, emd_cpu_nms
from misc_utils import ensure_dir, load_json_lines, save_json_lines
from megengine.core._imperative_rt.utils import Logger
import network, dataset
import pdb
Logger.set_log_level(Logger.LogLevel.Error)
def eval_all(model_file, records, args):
        
    assert osp.exists(model_file)
    # multiprocessing
    num_records = len(records)
    num_devs = args.devices
    num_image = math.ceil(num_records / num_devs)
    result_queue = Queue(10000)
    procs = []
    all_results = []
    pbar = tqdm(total = num_records, leave = False, ascii = True)
    for i in range(num_devs):
        start = i * num_image
        end = min(start + num_image, num_records)
        split_records = records[start:end]
        # inference(model_file, i, split_records, result_queue)
        proc = Process(target=inference, args=(
                model_file, i, split_records, result_queue))
        proc.start()
        procs.append(proc)
    
    for i in range(num_records):
        t = result_queue.get()
        
        if t is not None:
            all_results.append(t)
        pbar.update(1)
    
    for p in procs:
        p.join()
    
    return all_results

def load_model(net, model_file):

    assert osp.exists(model_file)
    checkpoint = mge.load(model_file)
    state_dict = checkpoint['state_dict']
    names = [k for k, _ in state_dict.items()]
    for name in names:
        if name.startswith('inputs.'):
            del state_dict[name]

    net.load_state_dict(state_dict, strict=False)
    return net

def inference(model_file, device, records, result_queue):

    def val_func():
        pred_boxes = net(net.inputs)
        return pred_boxes

    mge.set_default_device('xpu{}'.format(device))
    net = network.Network()
    net = load_model(net, model_file)
    net.eval()
    
    for record in records:

        np.set_printoptions(precision=2, suppress=True)
        image, gt_boxes, im_info, ID = get_data(record, device)

        net.inputs["image"].set_value(image.astype(np.float32))
        net.inputs["im_info"].set_value(im_info)
        del record, image

        pred_boxes = val_func().numpy()
        pred_bbox = pred_boxes[:, 1]
        scale = im_info[0, 2]
        cls_dets = pred_bbox[:, :4] / scale
        cls_dets = np.hstack([cls_dets, pred_bbox[:, 4:5]])
        if config.test_nms_version == 'set_nms':
            
            n = cls_dets.shape[0] // 2
            idents = np.tile(np.linspace(0, n-1, n).reshape(-1, 1),(1, 2)).reshape(-1, 1)
            cls_dets = np.hstack([cls_dets, idents])
            flag = cls_dets[:, 4] >= 0.05
            cls_dets = cls_dets[flag]
            keep = emd_cpu_nms(cls_dets, 0.5)
            
            cls_dets = cls_dets[keep].astype(np.float64)
            pred_tags = np.ones([cls_dets.shape[0],]).astype(np.float64)

        elif config.test_nms_version == 'nms':

            flag = cls_dets[:, 4] >= config.test_cls_threshold
            cls_dets = cls_dets[flag].astype(np.float32)
            keep = nms(cls_dets, config.test_nms)
            cls_dets = cls_dets[keep].astype(np.float64)
            pred_tags = np.ones([len(keep),]).astype(np.float64)

        else:
            raise ValueError('Unknown NMS method.')
        gt_boxes = gt_boxes.astype(np.float64)

        dtboxes = boxes_dump(cls_dets[:, :5], pred_tags, False)
        gtboxes = boxes_dump(gt_boxes, None, True)

        height, width = int(im_info[0, 3]), int(im_info[0, 4])
        result_dict = dict(ID=ID, height=height, width=width,
                dtboxes = dtboxes, gtboxes = gtboxes)
        result_queue.put_nowait(result_dict)

def boxes_dump(boxes, pred_tags, is_gt):
    
    result = []
    assert boxes.shape[1] >= 5
    boxes = boxes.tolist()
    for idx in range(len(boxes)):
        box = boxes[idx]
        if is_gt:
            box_dict = {}
            box_dict['box'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            box_dict['tag'] = box[-1]
        else:
            box_dict = {}
            box_dict['box'] = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            box_dict['tag'] = int(pred_tags[idx])
            box_dict['score'] = box[4]
        result.append(box_dict)
    return result

def get_data(record, device):

    data = dataset.val_dataset(record)
    image, gt_boxes, ID = \
                data['data'], data['boxes'], data['ID']
    if config.eval_resize == False:
        resized_img, scale = image, 1
    else:
        resized_img, scale = dataset.resize_img_by_short_and_max_size(
            image, config.eval_image_short_size, config.eval_image_max_size)

    original_height, original_width = image.shape[0:2]
    height, width = resized_img.shape[0:2]
    transposed_img = np.ascontiguousarray(
        resized_img.transpose(2, 0, 1)[None, :, :, :],
        dtype=np.float32)
    im_info = np.array([height, width, scale, original_height, original_width],
        dtype=np.float32)[None, :]
    return transposed_img, gt_boxes, im_info, ID

def run_test():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--resume_weights', '-r', default=None, type=str)
    parser.add_argument('--start_epoch', '-s',default = 30, type=int)
    parser.add_argument('--end_epoch','-e', default= 50, type=int)
    parser.add_argument('--devices', '-d', default=1, type=int)
    args = parser.parse_args()
    # eval_all(args)

    # model_path
    model_dir = config.model_dir
    eval_dir = config.eval_dir
    # misc_utils.ensure_dir(evalDir)
    ensure_dir(config.eval_dir)
    records = load_json_lines(config.eval_source)

    start_epoch, end_epoch = args.start_epoch, args.end_epoch
    for epoch in range(start_epoch, end_epoch):
        model_file = osp.join(model_dir, 'epoch-{}.pkl'.format(epoch))
        if not osp.exists(model_file):
            continue

        results = eval_all(model_file, records, args)
        
        fpath = osp.join(eval_dir, 'epoch-{}.human'.format(epoch))
        save_json_lines(results, fpath)
    # model_file = os.path.join(saveDir, 
    #         'epoch_{}.pkl'.format(args.resume_weights))
    # assert os.path.exists(model_file)
    # load data
    
if __name__ == '__main__':

    run_test()

