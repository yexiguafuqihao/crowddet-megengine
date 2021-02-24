import os
import os.path as osp
import bisect
import argparse
import multiprocessing as mp
import time
import numpy as np
from tqdm import tqdm
import megengine as mge
from megengine import distributed as dist
from megengine import optimizer as optim
import megengine.autodiff as autodiff
from megengine import jit
# import dataset
import network
from config import config as cfg
from dataset.CrowdHuman import CrowdHuman
from misc_utils import ensure_dir
from megengine.core._imperative_rt.utils import Logger
from megengine import data
import pdb

ensure_dir(cfg.output_dir)
logger = mge.get_logger(__name__)
log_path = osp.join(cfg.output_dir, 'logger.log')
mge.set_log_file(log_path, mode='a')
Logger.set_log_level(Logger.LogLevel.Error)

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
def allreduce_cb(param, grad, group=dist.WORLD):
    return dist.functional.all_reduce_sum(grad, group) / group.size

def train_one_epoch(model, gm, data_iter, opt, max_steps, rank, epoch_id, gpu_num):
    
    # @jit.trace(symbolic=False,)
    def propagate():
        with gm:
            loss_dict = model(model.inputs)
            total_loss = sum([loss_dict[key].mean() for key in loss_dict.keys()])
            gm.backward(total_loss)
            opt.step().clear_grad()
            loss_dict['total_loss'] = total_loss
        return loss_dict
    tic = time.time()
    workspace = osp.split(osp.realpath(__file__))[0]
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
        
        image, boxes, im_info = next(data_iter)
        model.inputs["image"].set_value(image)
        model.inputs["gt_boxes"].set_value(boxes)
        model.inputs["im_info"].set_value(im_info)
        m = image.shape[0]
        del image, boxes, im_info
        losses = propagate()

        print_str = ' '
        for loss_name, loss_value in losses.items():
            print_str += ', {}: {:.4f}'.format(loss_name, loss_value.numpy())
        
        if rank == 0:
            if step % cfg.log_dump_interval == 0:
                speed = cfg.log_dump_interval * m / (time.time() - tic)
                tic = time.time()
                logger.info(
                    "epoch-{}, {}/{}, speed:{:.3f} mb/s, lr: {:.4f}{}.\n{}".format(
                    epoch_id,
                    step,
                    max_steps,
                    speed,
                    opt.param_groups[0]["lr"],
                    print_str,
                    workspace,
                    )
                )

def worker(rank, gpu_num, args):
    # using sublinear
    os.environ["MGB_COMP_GRAPH_OPT"] = "enable_sublinear_memory_opt=1;seq_opt.enable_seq_comp_node_opt=0"
    os.environ["MGB_SUBLINEAR_MEMORY_GENETIC_NR_ITER"] = '10'
    os.environ['MGB_CUDA_RESERVE_MEMORY'] = '1'
    # establish the server if is the master

    dist_port = args.port
    if rank == 0:
        dist.Server(port=dist_port)
    if gpu_num> 1:

        dist.init_process_group(
            master_ip="localhost",
            port=dist_port,
            world_size=gpu_num,
            rank=rank,
            device=rank,
        )
        logger.info("Init process group for gpu%d done", rank)

    model = network.Network()
    params = model.parameters(requires_grad=True)
    model.train()

    # Autodiff gradient manager
    gm = autodiff.GradManager().attach(
        model.parameters(),
        callbacks=allreduce_cb,
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

    start_epoch = 0
    if args.resume_weights is not None:
        assert osp.exists(args.resume_weights)
        model_file = args.resume_weights
        print('Loading {} to initialize FPN...'.format(model_file))
        model_dict = mge.load(model_file)
        start_epoch, weights = model_dict['epoch'] + 1, model_dict['state_dict']
        model.load_state_dict(weights, strict=False)
    
    logger.info("Prepare dataset")
    # train_loader = dataset.train_dataset(rank)

    train_dataset = CrowdHuman(cfg, if_train=True)
    train_sampler = data.Infinite(data.RandomSampler(
        train_dataset, batch_size = cfg.batch_per_gpu, drop_last=True,
        world_size = gpu_num, rank = rank,))
    train_loader = data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        collator = train_dataset,
        num_workers=4,
    )
    
    train_loader = iter(train_loader)
    logger.info("Training...")
    for epoch_id in range(start_epoch, cfg.max_epoch):
        for param_group in opt.param_groups:
            param_group["lr"] = (
                cfg.basic_lr * gpu_num * cfg.batch_per_gpu
                * (cfg.lr_decay_rate ** bisect.bisect_right(cfg.lr_decay_sates, epoch_id))
            )

        max_steps = cfg.nr_images_epoch // (cfg.batch_per_gpu * gpu_num)
        train_one_epoch(model, gm, train_loader, opt, max_steps, rank, epoch_id, gpu_num)
        if rank == 0:
            save_path = osp.join(cfg.model_dir, 'epoch-{}.pkl'.format(epoch_id + 1))
            state_dict = model.state_dict()
            names = [k for k, _ in state_dict.items()]
            for name in names:
                if name.startswith('inputs.'):
                    del state_dict[name]

            mge.save(
                {"epoch": epoch_id, "state_dict": state_dict}, save_path,
            )
            logger.info("dump weights to %s", save_path)

def train(args):
    # ------------------------ begin training -------------------------- #
    valid_nr_dev = mge.get_device_count("gpu")
    gpu_num = min(valid_nr_dev, args.num_gpus)
    assert gpu_num > 0
    logger.info('Device Count: {}'.format(gpu_num))

    ensure_dir(cfg.model_dir)

    if not osp.exists('output'):
        os.symlink(cfg.output_dir,'output')

    if gpu_num > 1:
        args.port =find_free_port()
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
    parser.add_argument('--port', '-pt', type=int, default=11123)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    run_train()

