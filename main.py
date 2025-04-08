import os
import copy
import argparse
import warnings

import yaml
import torch
import numpy as np
from tqdm import tqdm
from timm import utils
from loguru import logger
from torch.utils import data
import torchvision.models as models

from utils import util
from utils.dataset import Dataset
from nets.PIPNetLandmarksDetector import LandmarksDetectorONNX
from nets.PIPNetLandmarksDetectorPT import LandmarksDetectorPT


warnings.filterwarnings("ignore")

util.setup_multi_processes()
util.init_deterministic_seed()


def lr(args, base_global_batch_size=256):
    """ Adjust learning rate for stable paper reproducing """
    return args.fixed_lr or (args.base_lr * args.batch_size * args.world_size / base_global_batch_size)


def create_result_files(args):
    """ Create all ouput results in runs/exp/ """
    util.safe_yaml_config_file(args)


def train(args, config):
    # Initialize model
    model = models.get_model(name=args.model_name, weights=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 12*2)
    
    # Initialize the regression layer
    torch.nn.init.normal_(model.fc.weight, std=0.001)
    if model.fc.bias is not None:
        torch.nn.init.constant_(model.fc.bias, 0)
    model.to(args.device)

    accumulate = 1 or max(1, round(256 / (args.batch_size * args.world_size)))  # FIXME

    # Optimizer
    optimizer = torch.optim.Adam(util.weight_decay(model), lr=args.lr)

    # Scheduler
    scheduler = util.CosineLR(args, optimizer)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None
    
    # Dataset
    dataset = Dataset(config, evaluate=False)
    
    sampler = None
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
                             num_workers=8, pin_memory=True, drop_last=True)

    if args.distributed:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    if args.local_rank == 0:
        print(f"GPUs: {args.world_size}, Accumulate: {accumulate}, LR: {args.lr}\n")
    
    # Start training
    best_NME = float('inf')
    num_batch = len(loader)
    performance_results = []
    training_results_csv = []
    actual_learning_rate = []
    criterion = util.ComputeLossRegg()
    amp_scale = torch.amp.GradScaler("cuda")

    # Training loop
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']

        p_bar = enumerate(loader)
        m_loss = util.AverageMeter()

        if args.distributed:
            sampler.set_epoch(epoch)

        if args.local_rank == 0:
            print(f"{'epoch':>10}{'loss':>10}")
            p_bar = tqdm(iterable=p_bar, total=num_batch)

        model.train()
        optimizer.zero_grad()

        # Loading Batch Data:
        for i, (samples, targets) in p_bar:
            samples = samples.to(args.device)
            targets = targets.to(args.device)

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(samples)
                loss = criterion(outputs, targets)
                loss = loss / accumulate        # Scale loss by accumulate to prevent gradient explosion

            # Backward pass with scaled loss
            amp_scale.scale(loss).backward()

            # Perform optimizer step after accumulating gradients for 'accumulate' batches
            if (i % accumulate) == 0:
                amp_scale.step(optimizer)       # Update model weights
                amp_scale.update()              # Update scaler for next iteration
                optimizer.zero_grad()           # Reset gradients for the next accumulation
                if ema:
                    ema.update(model)           # Update EMA model if used

            # Synchronize GPUs
            torch.cuda.synchronize()
            
            # Reduce loss for distributed mode
            if args.distributed:
                loss = utils.reduce_tensor(loss.data, args.world_size)

            # Update running average of the loss
            m_loss.update(loss.item(), samples.size(0))
            
            # Log progress on the main process
            if args.local_rank == 0:
                s = ('%10s' + '%10.4g') % (f'{epoch + 1}/{args.epochs}', m_loss.avg)
                p_bar.set_description(s)

        # Scheduler
        scheduler.step(epoch+1)

        if args.local_rank == 0:
            # Calculate test NME
            last_NME = test(args, config, copy.deepcopy(ema.ema))

            # Save results to graph
            result = (epoch+1, float(m_loss.avg), float(last_NME))

            performance_results.append(result)
            util.plot_performance_results(args, performance_results)

            actual_learning_rate.append((epoch+1, float(current_lr)))
            util.plot_actual_learning_rate(args, actual_learning_rate)

            # Save results to csv file
            epoch_csv_results = {
                "epoch":            str(epoch+1).zfill(3),
                "train_loss":       str(f'{m_loss.avg:.5f}'),
                "lr":               str(f'{current_lr:.8f}'),
                'NME':              str(f'{last_NME:.3f}'),
                'status':           'Best NME' if best_NME > last_NME else 'passed',
            }
            training_results_csv.append(epoch_csv_results)
            util.save_results_csv(args, training_results_csv)
            
            # Save last model
            ckpt = {'model': copy.deepcopy(ema.ema).half()}
            torch.save(ckpt, f"{args.output_dir}/weights/last.pt")

            # Update best NME
            if best_NME > last_NME:
                best_NME = last_NME

                # Save best pt model
                torch.save(ckpt, f"{args.output_dir}/weights/best.pt")

                # Save best onnx model
                model_onnx = copy.deepcopy(ema.ema).float().eval()
                if hasattr(model_onnx, 'fuse'): model_onnx = model_onnx.fuse()
                example_input = torch.randn(1, 3, config["input_size"], config["input_size"]).to(args.device)
                util.export_onnx_model(args, model_onnx, example_input, f"{args.output_dir}/weights/best.onnx")
                
            del ckpt

    if args.local_rank == 0:
        util.strip_optimizer(f"{args.output_dir}/weights/best.pt")  # strip optimizers
        util.strip_optimizer(f"{args.output_dir}/weights/last.pt")  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, config, model=None):
    # Get model path
    model_path = util.find_model_path(args, extension="pt")

    # Load DataLoader
    loader = data.DataLoader(Dataset(config, evaluate=True))

    # Load PT model
    pt_model = LandmarksDetectorPT(config,
                                   model_path,
                                   device=args.device,
                                   half=True).load_model(model)
    
    nme_merge = []
    for sample, target in tqdm(loader, '%20s' % 'NME'):
        nme = pt_model.test(sample, target)
        nme_merge.append(nme)

    # Print results
    nme = np.mean(nme_merge) * 100
    print(f"{nme:20.5g}\n")
    return nme


def test_onnx(args, config):
    # Get model path
    model_path = util.find_model_path(args, extension="onnx")

    # Load DataLoader
    loader = data.DataLoader(Dataset(config, evaluate=True))

    # Load ONNX model
    onnx_model = LandmarksDetectorONNX(config,
                                       model_path,
                                       apply_all_optim=True,
                                       device="cup").load_model()

    nme_merge = []
    for sample, target in tqdm(loader, '%20s' % 'ONNX-NME'):
        nme = onnx_model.test(sample, target)
        nme_merge.append(nme)

    # Print results
    nme = np.mean(nme_merge) * 100
    print(f"{nme:20.5g}\n")
    return nme


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',         type=str,   default="regnet_x_800mf")
    parser.add_argument('--epochs',             type=int,   default=120)
    parser.add_argument('--batch-size',         type=int,   default=256)
    parser.add_argument('--input-size',         type=int,   default=192)
    parser.add_argument('--lr',                 type=float, default=0.0005)
    parser.add_argument('--exp',                type=str,   default=None)

    parser.add_argument('--device',             type=str,   default="cuda")
    parser.add_argument('--opset-version',      type=int,   default=11)
    parser.add_argument('--output-dir',         type=str,   default=None)

    parser.add_argument('--train',              action='store_true')
    parser.add_argument('--test',               action='store_true')
    parser.add_argument('--test-onnx',          action='store_true')

    parser.add_argument('--comment',            type=str,   default=None,   required=False,
                        help='teamCode-projectName-taskName-modelVersion')
    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    with open(os.path.join('utils', 'config.yaml'), errors='ignore') as f:
        config = yaml.safe_load(f)
        config["input_size"] = args.input_size
        args.config = {k: v for k, v in config.items() if "indices" not in k}

    # Initialize experiment space
    if args.train:
        args.output_dir = util.set_experiment_results_output(args)

    if args.train and args.local_rank == 0:
        create_result_files(args)
        save_path = os.path.join(args.output_dir, "weights")
        os.makedirs(save_path, exist_ok=True)
        os.makedirs('weights', exist_ok=True)

    if args.train:
        train(args, config)
    if args.test and args.local_rank == 0:
        nme = test(args, config)
    if args.test_onnx and args.local_rank == 0:
        nme_onnx = test_onnx(args, config)
    if args.train and args.test and args.test_onnx and args.local_rank == 0:
        util.write_experiment_results_storage(args, nme, nme_onnx)


if __name__ == "__main__":
    main()
