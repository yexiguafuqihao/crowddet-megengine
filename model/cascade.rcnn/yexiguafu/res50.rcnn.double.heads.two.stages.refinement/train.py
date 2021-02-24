import os
import os.path as osp
import bisect
import argparse
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
import megengine as mge
from megengine import distributed as dist
from megengine import optimizer as optim
# from megengine.autodiff.grad_manager import GradManager
import megengine.autodiff as autodiff
from megengine import jit
import dataset
import network
from config import config as cfg
import pdb

logger = mge.get_logger(__name__)
# log_path = os.path.join(cfg.output_dir, 'net_log.log')
log_path = osp.join(cfg.output_dir, 'logger.log')
mge.set_log_file(log_path, mode='a')

def allreduce_cb(param, grad, group=dist.WORLD):
    return dist.functional.all_reduce_sum(grad, group) / group.size

def train_one_epoch(model, gm, data_iter, opt, max_steps, rank, epoch_id, gpu_num):
    
    # @jit.trace(symbolic=True)
    def propagate():
        with gm:
            loss_dict = model(model.inputs)
            total_loss = sum([loss_dict[key].mean() for key in loss_dict.keys()])
            pdb.set_trace()
            gm.backward(total_loss)
            opt.step().clear_grad()
            loss_rpn_cls = loss_dict['loss_rpn_cls']
            loss_rpn_loc = loss_dict['loss_rpn_loc']
            loss_rcnn_emd = loss_dict['loss_rcnn_emd']
        return (total_loss, loss_rpn_cls, loss_rpn_loc,
                loss_rcnn_emd)
    
    for step in range(max_steps):
        # learing rate
        if epoch_id == 0 and step < cfg.warm_iters:
            base_lr = (
                cfg.basic_lr * gpu_num * cfg.batch_per_gpu
                * (cfg.lr_decay_rate ** bisect.bisect_right(cfg.lr_decay_sates, epoch_id)
                )
            )
            lr_factor = (step + 1.0) / cfg.warm_iters
            for param_group in opt.param_groups:
                param_group["lr"] = 0.33 * base_lr + 0.67 * lr_factor * base_lr
        mini_batch = next(data_iter)
        im_info = mini_batch["im_info"]
        image = mini_batch["data"][:, :, :int(im_info[0, 0]), :int(im_info[0, 1])]
        model.inputs["image"].set_value(image)
        model.inputs["gt_boxes"].set_value(mini_batch["boxes"])
        model.inputs["im_info"].set_value(mini_batch["im_info"])
        opt.zero_grad()
        losses = propagate()
        opt.step()

        if rank == 0:
            if step % cfg.log_dump_interval == 0:
                logger.info(
                    "e%d, %d/%d, lr:%f, total_loss:%f, rpn_cls:%f, rpn_loc:%f, rcnn_emd:%f",
                    epoch_id,
                    step,
                    max_steps,
                    opt.param_groups[0]["lr"],
                    losses[0].numpy(),
                    losses[1].numpy(),
                    losses[2].numpy(),
                    losses[3].numpy(),
                )

def worker(rank, gpu_num, args):
    # using sublinear
    os.environ["MGB_COMP_GRAPH_OPT"] = "enable_sublinear_memory_opt=1;seq_opt.enable_seq_comp_node_opt=0"
    os.environ["MGB_SUBLINEAR_MEMORY_GENETIC_NR_ITER"] = '50'
    if gpu_num> 1:
        dist.init_process_group(
            master_ip="localhost",
            master_port=23456,
            world_size=gpu_num,
            rank=rank,
            dev=rank,
        )
        logger.info("Init process group for gpu%d done", rank)

    model = network.Network()
    params = model.parameters(requires_grad=True)
    model.train()

    # Autodiff gradient manager
    gm = autodiff.GradManager().attach(
        model.parameters(),
        callbacks=allreduce_cb,
        # callbacks=dist.make_allreduce_cb("MEAN"),
    )

    opt = optim.SGD(
        params,
        lr=cfg.basic_lr * gpu_num * cfg.batch_per_gpu,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    if cfg.pretrain_weight is not None:
        weights = mge.load(cfg.pretrain_weight)
        del weights['fc.weight']
        del weights['fc.bias']
        model.resnet50.load_state_dict(weights)
    
    logger.info("Prepare dataset")
    train_loader = dataset.train_dataset(rank)
    logger.info("Training...")
    for epoch_id in range(cfg.max_epoch):
        for param_group in opt.param_groups:
            param_group["lr"] = (
                cfg.basic_lr * gpu_num * cfg.batch_per_gpu
                * (cfg.lr_decay_rate ** bisect.bisect_right(cfg.lr_decay_sates, epoch_id))
            )

        max_steps = cfg.nr_images_epoch // (cfg.batch_per_gpu * gpu_num)
        train_one_epoch(model, gm, train_loader, opt, max_steps, rank, epoch_id, gpu_num)
        if rank == 0:
            save_path = os.path.join(cfg.model_dir, 'epoch_{}.pkl'.format(epoch_id + 1))
            mge.save(
                {"epoch": epoch_id, "state_dict": model.state_dict()}, save_path,
            )
            logger.info("dump weights to %s", save_path)

def train(args):
    # ------------------------ begin training -------------------------- #
    valid_nr_dev = mge.get_device_count("gpu")
    # if args.divice_num == -1:
    #     gpu_num = valid_nr_dev
    # else:
    #     if args.divice_num > valid_nr_dev:
    #         logger.error("do not have enough gpus for training")
    #         sys.exit(1)
    #     else:
    #         gpu_num = args.divice_num
    gpu_num = min(valid_nr_dev, args.num_gpus)
    # logger.info("Device Count = %d", gpu_num)
    assert gpu_num > 0
    logger.info('Device Count: {}'.format(gpu_num))

    model_dir = cfg.model_dir
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if gpu_num > 1:
        mp.set_start_method("spawn")
        processes = list()
        for i in range(gpu_num):
            process = mp.Process(target=worker, args=(i, gpu_num, args))
            process.start()
            processes.append(process)

        for p in processes:
            p.join()
    else:
        worker(0, 1, args)

def run_train():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_gpus", "-d", default=-1, type=int, help="total number of gpus for training")
    parser.add_argument('--resume_weights', '-r', default=None, type=str)
    parser.add_argument('--progressbar', '-p', action='store_true', default=False)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    run_train()

