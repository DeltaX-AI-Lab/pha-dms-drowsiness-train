import os
import sys
import csv
import copy
import math
import random
from pathlib import Path
from platform import system
from datetime import datetime

import cv2
import yaml
import onnx
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger
from onnxsim import simplify
from PIL import Image, ImageFilter


def init_deterministic_seed(seed=0, deterministic_mode=True):
    """ Setup random seed """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = not deterministic_mode
    torch.backends.cudnn.deterministic = deterministic_mode


def setup_multi_processes():
    """	Setup multi-processing environment variables """

    # Set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # Disable opencv multithreading to avoid system being overloaded (incompatible with PyTorch DataLoader)
    cv2.setNumThreads(0)

    # Setup OMP threads
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '16'

    # Setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = '16'

    if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_experiment_results_output(args, root="runs"):
	experiment_path = ""

	if args.output_dir is not None:
		root = args.output_dir

	if os.path.exists(root):
		exps = os.listdir(root)
		exps_index = [int(exp.split("_")[-1]) for exp in exps if exp.startswith("exp_") and exp.split("_")[-1].isdigit()]
		next_index = max(exps_index) + 1 if exps_index else 0
		experiment_path = os.path.join(root, f"exp_{next_index:03d}")
	else:
		experiment_path = os.path.join(root, f"exp_{0:03d}")

	if args.local_rank == 0:
		os.makedirs(experiment_path, exist_ok=True)
	return experiment_path


def write_experiment_results_storage(args, nme, nme_onnx, csv_file_name="storage.csv"):
    # Get save directory and experiment name
    parts = args.output_dir.split("/")
    save_dir = "/".join(parts[:-1])
    
    exp = parts[-1].split("_")[-1]
    save_path = os.path.join(save_dir, csv_file_name)
    
    # Prepare the current datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    # Define the CSV file's headers
    headers = ['exp', 'nme', 'nme-onnx', 'date-time', 'depth']
    
    csv_results = {
        "exp":       str(exp).zfill(3),
        "nme":       str(nme),
        "nme-onnx":  str(nme_onnx),
        'date-time': str(current_datetime),
        'depth':     str(args.depth),
    }

    # Check if the file already exists
    file_exists = os.path.isfile(save_path)
    
    # Writing to the CSV file
    with open(save_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_results)
        file.flush()


def safe_yaml_config_file(args):
    """ Save the training config file in yaml format """
    clean_args = vars(args)
    with open(os.path.join(args.output_dir, "training_config_pipeline.yaml"), 'w') as yaml_file:
        yaml.dump(clean_args, yaml_file, default_flow_style=False)


def save_results_csv(args, training_results, csv_file_name="training_results.csv"):
    save_path = os.path.join(args.output_dir, csv_file_name)

    # Define the CSV file's headers
    headers = ['epoch', 'train_loss', 'lr', 'NME', 'status']

    # Writing to the CSV file
    with open(save_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for result in training_results:
            writer.writerow(result)
        file.flush()


def setup_logger(log_name='exp'):
    """ Setup a logger environments for different purposes

    LEVELS = [TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL]
    show messages:
        logger.trace("A trace message.")
        logger.debug("A debug message.")
        logger.info("An info message.")
        logger.success("A success message.")
        logger.warning("A warning message.")
        logger.error("An error message.")
        logger.critical("A critical message.")

        colorize=None --> the choice is automatically made based on the sink being a tty or not.
    """
    folder = "loggers"
    Path(folder).mkdir(parents=True, exist_ok=True)
    cur_date_time = datetime.now().strftime("%d.%m.%Y-%H-%M-%S")

    # For terminal - configuration to stderr (Optionally)
    logger.remove(0)    # To remove default version of logger
    default_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(sink=sys.stderr, level='INFO', format=default_format, filter=None, colorize=None, serialize=False, backtrace=True, diagnose=True, enqueue=False, context=None, catch=True)

    # For logger file - configuration
    log_path = os.path.join(folder, f"{log_name}_{cur_date_time}.log")
    log_format = "{time:YYYY-MM-DD HH:mm:ss} | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sink=log_path, level="TRACE", format=log_format, colorize=None, rotation="10 MB")


def set_experiment_logger(args):
    """ Create a logger file in experiment logger """
    cur_date_time = datetime.now().strftime("%d.%m.%Y-%H-%M-%S")
    log_path = os.path.join(args.output_dir, f"{cur_date_time}.log")
    log_format = "{time:YYYY-MM-DD HH:mm:ss} | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(sink=log_path, level="TRACE", format=log_format, colorize=None, rotation="10 MB")


def find_model_path(args, extension="onnx"):
    """ Find model path to test the model """
    if args.train:                              # by default train path
        model_path = f"{args.output_dir}/weights/best.{extension}"
    elif args.exp is None:                      # by --exp None
        print("Error: Missing argument. Please provide --exp with a model path or an experiment ID."); exit()
    elif args.exp.isdigit():                    # by --exp N
        model_path = f"runs/exp_{args.exp}/weights/best.{extension}"
    elif args.exp.endswith(f".{extension}"):    # by --exp absolute path
        model_path = args.exp
    else:
        exit()
    return model_path


def weight_decay(model, decay=5E-5):
    p1 = []
    p2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            p1.append(param)
        else:
            p2.append(param)
    return [{'params': p1, 'weight_decay': 0.}, {'params': p2, 'weight_decay': decay}]


def plot_lr(args, optimizer, scheduler):
    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        y.append(optimizer.param_groups[0]['lr'])
        scheduler.step(epoch + 1)

    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, args.epochs)
    plt.ylim(0)
    plt.savefig('./weights/lr.png', dpi=200)
    plt.clf()
    plt.close()


def plot_performance_results(args, performance_results):
    save_path = os.path.join(args.output_dir, f"performance_results.svg")
    
    # Extract epoch numbers, accuracies, and losses for training and validation
    epochs, train_losses, nme = zip(*performance_results)

    best_test_nme = min(nme)
    stop_epoch_nme = nme.index(best_test_nme) + 1

    best_train_loss = min(train_losses)
    stop_epoch_train_loss = train_losses.index(best_train_loss) + 1

    # Title
    results = "Epoch={0}/{1}, train-loss={2:.4f}, test-NME={3:.5f}".format(
        stop_epoch_train_loss, stop_epoch_nme, best_train_loss, best_test_nme)
    
    # Initialize the plot
    fig, ax1 = plt.subplots(figsize=(20, 12))
    plt.title(f'Results: {results}', fontsize=20)
    plt.xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('NME', fontsize=16)

    # Plot the validation NME
    ax1.plot(epochs, nme, label='NME', marker='o')
    ax1.tick_params(axis='y')

    ax1.scatter(x=stop_epoch_nme, y=best_test_nme, s=50, color='k', zorder=5.5)
    ax1.axhline(y=best_test_nme, color='k', linestyle='--', linewidth=2.0)
    ax1.axvline(x=stop_epoch_nme, color='k', linestyle='--', linewidth=2.0)

    # Create a second y-axis for the loss
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', fontsize=16, color='r')
    ax2.plot(epochs, train_losses, color='r', linestyle='--', marker='o', label='Train Loss', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor='r')

    ax2.scatter(x=stop_epoch_train_loss, y=best_train_loss, s=50, color='k', zorder=5.5)
    ax2.axhline(y=best_train_loss, color='k', linestyle='--', linewidth=2.0)
    ax2.axvline(x=stop_epoch_train_loss, color='r', linestyle='--', linewidth=1.0, alpha=0.6)

    # Show grid, legend, and labels
    ax1.grid(True)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')

    plt.savefig(save_path, dpi=600)
    plt.clf()
    plt.close()


def plot_performance_results_two_losses(args, performance_results):
    save_path = os.path.join(args.output_dir, f"performance_results.svg")
    
    # Extract epoch numbers, accuracies, and losses for training and validation
    epochs, train_losses, test_losses, nme = zip(*performance_results)

    best_test_nme = min(nme)
    stop_epoch_nme = nme.index(best_test_nme) + 1

    best_train_loss = min(train_losses)
    stop_epoch_train_loss = train_losses.index(best_train_loss) + 1

    best_test_loss = min(test_losses)
    stop_epoch_test_loss = test_losses.index(best_test_loss) + 1

    # Title
    results = "Epoch={0}/{1}, train-loss={2:.4f}, test-loss={3:.4f}, test-NME={4:.5f}".format(
        stop_epoch_nme, stop_epoch_test_loss, best_train_loss, best_test_loss, best_test_nme)
    
    # Initialize the plot
    fig, ax1 = plt.subplots(figsize=(20, 12))
    plt.title(f'Results: {results}', fontsize=20)
    plt.xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('NME', fontsize=16)

    # Plot the validation NME
    ax1.plot(epochs, nme, label='NME', marker='o')
    ax1.tick_params(axis='y')

    ax1.scatter(x=stop_epoch_nme, y=best_test_nme, s=50, color='k', zorder=5.5)
    ax1.axhline(y=best_test_nme, color='k', linestyle='--', linewidth=2.0)
    ax1.axvline(x=stop_epoch_nme, color='k', linestyle='--', linewidth=2.0)

    # Create a second y-axis for the loss
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', fontsize=16, color='r')
    ax2.plot(epochs, train_losses, color='b', linestyle='--', marker='o', label='Train Loss', alpha=0.6)
    ax2.plot(epochs, test_losses, color='r', linestyle='--', marker='o', label='Test Loss')
    ax2.tick_params(axis='y', labelcolor='r')

    ax2.scatter(x=stop_epoch_test_loss, y=best_test_loss, s=50, color='k', zorder=5.5)
    ax2.axhline(y=best_test_loss, color='k', linestyle='--', linewidth=2.0)
    ax2.axvline(x=stop_epoch_test_loss, color='r', linestyle='--', linewidth=1.0, alpha=0.6)

    # Show grid, legend, and labels
    ax1.grid(True)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center right')

    plt.savefig(save_path, dpi=600)
    plt.clf()
    plt.close()


def plot_actual_learning_rate(args, results):
    save_path = os.path.join(args.output_dir, f"actual_learning_rate.svg")
    
    # Extract epoch numbers, accuracies, and losses for training and validation
    epochs, learning_rates = zip(*results)

    # Initialize the plot
    plt.figure(figsize=(20, 12))
    plt.title('Actual learning rate', fontsize=20)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('LR', fontsize=16)

    # Plot the training and validation accuracies
    plt.plot(epochs, learning_rates, label='LR scheduler', marker='o')

    # Show grid, legend, and labels
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=600)
    plt.clf()
    plt.close()


def strip_optimizer(filename):
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half().eval()   # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def load_weight(ckpt, model):
    dst = model.state_dict()
    src = torch.load(ckpt, map_location=torch.device('cpu'))['model'].float().state_dict()

    ckpt = {}
    for k, v in src.items():
        if k in dst and v.shape == dst[k].shape:
            ckpt[k] = v
    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def symplify_onnx_model(onnx_path):
    # Load your predefined ONNX model
    model = onnx.load(onnx_path)

    # Convert model
    model_simplified, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"

    # Save simplified model
    onnx.save(model_simplified, onnx_path)


def check_onnx_model(onnx_path, verbose=False):
    # Load the ONNX model
    model = onnx.load(onnx_path)

    # Check that the model is well formed
    onnx.checker.check_model(model, full_check=True)

    # Print a human readable representation of the graph
    if verbose: print(onnx.helper.printable_graph(model.graph))


def export_onnx_model(args, model, example_input, onnx_path, simplify=True, verify=True, only_arc=False):
    """ Save the model in ONNX format """
    export_params = True
    do_constant_folding = True
    input_names = ["input"]
    output_names = ["score", "offset_x", "offset_y", "neighbor_x", "neighbor_y"] if only_arc else ["output"]
    dynamic_axes = None
    verbose = False

    if dynamic_axes:
        dynamic_axes = {
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        }

    # Export the PyTorch model to ONNX
    torch.onnx.export(
        model=model,								# model being run
        args=example_input,							# model input (or a tuple for multiple inputs)
        f=onnx_path,								# where to save the model
        export_params=export_params,				# store the trained parameter weights inside the model
        opset_version=args.opset_version,			# the ONNX version to export the model to
        do_constant_folding=do_constant_folding,	# to execute constant folding for optimization
        input_names=input_names,					# specify the names of input
        output_names=output_names,					# specify the names of output
        verbose=verbose,							# prints a description of the model being exported to stdout
        dynamic_axes=dynamic_axes					# variable length axes
    )

    if simplify: symplify_onnx_model(onnx_path)
    if verify: check_onnx_model(onnx_path)


def resample():
    return random.choice((Image.NEAREST, Image.BILINEAR, Image.BICUBIC))


class RandomGaussianBlur:
    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            radius = random.random() * 5
            gaussian_blur = ImageFilter.GaussianBlur(radius)
            image = image.filter(gaussian_blur)
        return image, label


class RandomSandEffect:
    def __init__(self, density=0.02, intensity=255, grain_size=(1, 3), p=0.5):
        """
        Randomly applies a sand effect to an image by overlaying noise resembling sand grains.

        Args:
            density (float): Fraction of pixels to be turned into sand (e.g., 0.02 for 2%).
            intensity (int): Intensity of sand grains (0-255).
            grain_size (tuple): Range of sand grain sizes in pixels (min, max).
            p (float): Probability of applying the augmentation.
        """
        self.density = density
        self.intensity = intensity
        self.grain_size = grain_size
        self.p = p

    def __call__(self, image, label):
        """
        Apply the sand effect to the image with a given probability.

        Args:
            image (PIL.Image.Image): Input image.
            label: Corresponding label (not modified).

        Returns:
            image (PIL.Image.Image): Augmented image.
            label: Unchanged label.
        """
        if random.random() < self.p:
            # Convert PIL image to NumPy array
            image = np.array(image)

            # Ensure image is in BGR format for OpenCV
            if len(image.shape) == 2:  # Grayscale image
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA image
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

            # Get image dimensions
            height, width, _ = image.shape

            # Calculate the number of sand grains
            num_grains = int(self.density * height * width)

            # Generate random positions and add sand grains
            for _ in range(num_grains):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                grain_diameter = np.random.randint(self.grain_size[0], self.grain_size[1] + 1)
                cv2.circle(image, (x, y), grain_diameter, (self.intensity, self.intensity, self.intensity), -1)

            # Convert back to PIL image
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return image, label


class RandomHSV:
    def __init__(self, h=0.015, s=0.7, v=0.4, p=0.5):
        self.h = h
        self.s = s
        self.v = v
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = np.array(image)
            r = np.random.uniform(-1, 1, 3)
            r = r * [self.h, self.s, self.v] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype('uint8')
            lut_sat = np.clip(x * r[1], 0, 255).astype('uint8')
            lut_val = np.clip(x * r[2], 0, 255).astype('uint8')

            image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB, dst=image)
            image = Image.fromarray(image)
        return image, label


class RandomRGB2IR:
    """
    RGB to IR conversion
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() > self.p:
            return image, label
        image = np.array(image)
        image = image.astype('int32')
        delta = np.random.randint(10, 90)

        ir = image[:, :, 2]     # FIXME RGB:0 or BGR:2
        ir = np.clip(ir + delta, 0, 255)
        return Image.fromarray(np.stack((ir, ir, ir), axis=2).astype('uint8')), label


class RandomFlip:
    def __init__(self, flip_indices, p=0.5):
        self.flip_index = (np.array(flip_indices) - 1).tolist()
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = np.array(label).reshape(-1, 2)
            label = label[self.flip_index, :]
            label[:, 0] = 1 - label[:, 0]
            label = label.flatten()
            return image, label
        else:
            return image, label


class RandomTranslate:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            h, w = image.size
            a = 1
            b = 0
            c = int((random.random() - 0.5) * 60)
            d = 0
            e = 1
            f = int((random.random() - 0.5) * 60)
            image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample())
            label = label.copy()
            label = label.reshape(-1, 2)
            label[:, 0] -= 1. * c / w
            label[:, 1] -= 1. * f / h
            label = label.flatten()
            label[label < 0] = 0
            label[label > 1] = 1
            return image, label
        else:
            return image, label


class RandomRotate:
    def __init__(self, angle=45, p=0.5):
        self.angle = angle
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            num_lms = int(len(label) / 2)

            center_x = 0.5
            center_y = 0.5

            label = np.array(label) - np.array([center_x, center_y] * num_lms)
            label = label.reshape(num_lms, 2)
            theta = random.uniform(-np.radians(self.angle), +np.radians(self.angle))
            angle = np.degrees(theta)
            image = image.rotate(angle, resample=resample())

            cos = np.cos(theta)
            sin = np.sin(theta)
            label = np.matmul(label, np.array(((cos, -sin), (sin, cos))))
            label = label.reshape(num_lms * 2) + np.array([center_x, center_y] * num_lms)
            return image, label
        else:
            return image, label


class RandomCutOut:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = np.array(image).astype(np.uint8)
            image = image[:, :, ::-1]
            h, w, _ = image.shape
            cut_h = int(h * 0.4 * random.random())
            cut_w = int(w * 0.4 * random.random())
            x = int((w - cut_w - 10) * random.random())
            y = int((h - cut_h - 10) * random.random())
            image[y:y + cut_h, x:x + cut_w, 0] = int(random.random() * 255)
            image[y:y + cut_h, x:x + cut_w, 1] = int(random.random() * 255)
            image[y:y + cut_h, x:x + cut_w, 2] = int(random.random() * 255)
            image = Image.fromarray(image[:, :, ::-1].astype('uint8'), 'RGB')
            return image, label
        else:
            return image, label


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


class ComputeLossRegg:
    def __init__(self):
        super().__init__()
        self.criterion_reg = torch.nn.L1Loss()  # FIXME [torch.nn.L1Loss(), torch.nn.SmoothL1Loss()]
        self.criterion_cls_vis = torch.nn.BCEWithLogitsLoss()

    def __call__(self, outputs, targets):
        loss_reg = self.criterion_reg(outputs.float(), targets.float())
        return loss_reg


class ComputeLoss:
    def __init__(self, config):
        super().__init__()
        self.cls = config['cls']
        self.reg = config['reg']
        self.num_neighbor = config['num_nb']
        self.criterion_reg = torch.nn.L1Loss()  # FIXME [torch.nn.L1Loss(), torch.nn.SmoothL1Loss()]
        self.criterion_cls = torch.nn.MSELoss()

    def __call__(self, outputs, targets):
        device = outputs[0].device
        b, c, h, w = outputs[0].size()

        score = outputs[0]
        offset_x = outputs[1].view(b * c, -1)
        offset_y = outputs[2].view(b * c, -1)
        neighbor_x = outputs[3].view(b * self.num_neighbor * c, -1)
        neighbor_y = outputs[4].view(b * self.num_neighbor * c, -1)

        target_score = targets[0].to(device).view(b * c, -1)
        target_offset_x = targets[1].to(device).view(b * c, -1)
        target_offset_y = targets[2].to(device).view(b * c, -1)
        target_neighbor_x = targets[3].to(device).view(b * self.num_neighbor * c, -1)
        target_neighbor_y = targets[4].to(device).view(b * self.num_neighbor * c, -1)

        target_max_index = torch.argmax(target_score, 1).view(-1, 1)
        target_max_index_neighbor = target_max_index.repeat(1, self.num_neighbor).view(-1, 1)

        offset_x_select = torch.gather(offset_x, 1, target_max_index)
        offset_y_select = torch.gather(offset_y, 1, target_max_index)
        neighbor_x_select = torch.gather(neighbor_x, 1, target_max_index_neighbor)
        neighbor_y_select = torch.gather(neighbor_y, 1, target_max_index_neighbor)

        target_offset_x_select = torch.gather(target_offset_x, 1, target_max_index)
        target_offset_y_select = torch.gather(target_offset_y, 1, target_max_index)
        target_neighbor_x_select = torch.gather(target_neighbor_x, 1, target_max_index_neighbor)
        target_neighbor_y_select = torch.gather(target_neighbor_y, 1, target_max_index_neighbor)

        loss_cls = self.criterion_cls(score, target_score.view(b, c, h, w))
        loss_offset_x = self.criterion_reg(offset_x_select, target_offset_x_select)
        loss_offset_y = self.criterion_reg(offset_y_select, target_offset_y_select)
        loss_neighbor_x = self.criterion_reg(neighbor_x_select, target_neighbor_x_select)
        loss_neighbor_y = self.criterion_reg(neighbor_y_select, target_neighbor_y_select)

        loss_cls = self.cls * loss_cls
        loss_reg = self.reg * (loss_offset_x + loss_offset_y + loss_neighbor_x + loss_neighbor_y)
        return loss_cls + loss_reg


class ComputeLossGSSL:
    def __init__(self, config):
        super().__init__()
        self.cls = config['cls']
        self.reg = config['reg']
        self.num_neighbor = config['num_nb']
        self.criterion_reg = torch.nn.L1Loss()  # FIXME torch.nn.SmoothL1Loss()
        self.criterion_cls = torch.nn.MSELoss()

    def __call__(self, outputs, targets):
        device = outputs[0].device
        b, c, h, w = outputs[0].size()

        score1 = outputs[0].view(b * c, -1)
        score2 = outputs[1].view(b * c, -1)
        score3 = outputs[2].view(b * c, -1)
        offset_x = outputs[3].view(b * c, -1)
        offset_y = outputs[4].view(b * c, -1)
        neighbor_x = outputs[5].view(b * self.num_neighbor * c, -1)
        neighbor_y = outputs[6].view(b * self.num_neighbor * c, -1)

        target_score1 = targets[0].to(device).view(b * c, -1)
        target_score2 = targets[1].to(device).view(b * c, -1)
        target_score3 = targets[2].to(device).view(b * c, -1)
        target_offset_x = targets[3].to(device).view(b * c, -1)
        target_offset_y = targets[4].to(device).view(b * c, -1)
        target_neighbor_x = targets[5].to(device).view(b * self.num_neighbor * c, -1)
        target_neighbor_y = targets[6].to(device).view(b * self.num_neighbor * c, -1)

        target_max_index = torch.argmax(target_score1, 1).view(-1, 1)
        target_max_index_neighbor = target_max_index.repeat(1, self.num_neighbor).view(-1, 1)

        offset_x_select = torch.gather(offset_x, 1, target_max_index)
        offset_y_select = torch.gather(offset_y, 1, target_max_index)
        neighbor_x_select = torch.gather(neighbor_x, 1, target_max_index_neighbor)
        neighbor_y_select = torch.gather(neighbor_y, 1, target_max_index_neighbor)

        target_offset_x_select = torch.gather(target_offset_x, 1, target_max_index)
        target_offset_y_select = torch.gather(target_offset_y, 1, target_max_index)
        target_neighbor_x_select = torch.gather(target_neighbor_x, 1, target_max_index_neighbor)
        target_neighbor_y_select = torch.gather(target_neighbor_y, 1, target_max_index_neighbor)

        score = torch.cat([score1, score2, score3], 1)
        target_score = torch.cat([target_score1, target_score2, target_score3], 1)

        loss_cls = self.criterion_cls(score, target_score)
        loss_offset_x = self.criterion_reg(offset_x_select, target_offset_x_select)
        loss_offset_y = self.criterion_reg(offset_y_select, target_offset_y_select)
        loss_neighbor_x = self.criterion_reg(neighbor_x_select, target_neighbor_x_select)
        loss_neighbor_y = self.criterion_reg(neighbor_y_select, target_neighbor_y_select)
    
        loss_cls = self.cls * loss_cls
        loss_reg = self.reg * (loss_offset_x + loss_offset_y + loss_neighbor_x + loss_neighbor_y)
        return loss_cls + loss_reg
    
    
class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.updates = updates                  # number of EMA updates
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA

        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))

        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module

        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class CosineLR:
    def __init__(self, args, optimizer):
        self.min = 1E-5
        self.max = 1E-4

        self.optimizer = optimizer

        self.epochs = args.epochs
        self.values = [param_group['lr'] for param_group in self.optimizer.param_groups]

        self.warmup_epochs = 5
        self.warmup_values = [(v - self.max) / self.warmup_epochs for v in self.values]

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.max

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            values = [self.max + epoch * value for value in self.warmup_values]
        else:
            epoch = epoch - self.warmup_epochs
            if epoch < self.epochs:
                alpha = math.pi * (epoch - (self.epochs * (epoch // self.epochs))) / self.epochs
                values = [self.min + 0.5 * (lr - self.min) * (1 + math.cos(alpha)) for lr in self.values]
            else:
                values = [self.min for _ in self.values]

        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


class DataGenerator:
    def __init__(self, data_dir, target_size=256):
        """ Generator 300W train/test datasets """
        self.data_dir = data_dir
        self.target_size = target_size
        self.data_dir_img = os.path.join(data_dir, 'images')
        self.datasets = {
            "test": ['helen/testset', 'lfpw/testset', 'ibug'],
            "train": ['afw', 'helen/trainset', 'lfpw/trainset'],
        }
        os.makedirs(os.path.join(self.data_dir_img, 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir_img, 'train'), exist_ok=True)

    def process_300W(self, folder_name, image_name, label_name, scale=1.1):
        """ Crop a face and recalculate its facial landmarks """
        image_path = os.path.join(self.data_dir, folder_name, image_name)
        label_path = os.path.join(self.data_dir, folder_name, label_name)

        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        
        with open(label_path, 'r') as f:
            annotation = f.readlines()[3:-1]
            annotation = [x.strip().split() for x in annotation]
            annotation = [[int(float(x[0])), int(float(x[1]))] for x in annotation]

            anno_x = [x[0] for x in annotation]
            anno_y = [x[1] for x in annotation]

            x_min = min(anno_x)
            y_min = min(anno_y)
            x_max = max(anno_x)
            y_max = max(anno_y)
            box_w = x_max - x_min
            box_h = y_max - y_min
            
            x_min -= int((scale - 1) / 2 * box_w)
            y_min -= int((scale - 1) / 2 * box_h)
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)

            box_w = int(box_w * scale)
            box_h = int(box_h * scale)
            box_w = min(box_w, image_width - x_min - 1)
            box_h = min(box_h, image_height - y_min - 1)

            x_max = x_min + box_w
            y_max = y_min + box_h
            image_crop = image[y_min:y_max, x_min:x_max, :]
            image_crop = cv2.resize(image_crop, (self.target_size, self.target_size))
            annotation = [[(x - x_min) / box_w, (y - y_min) / box_h] for x, y in annotation]
            return image_crop, annotation

    def process_dataset(self, subset):
        """ Preprocess 300W train/test datasets """
        print(f"======== {subset.capitalize()} Dataset ========")
        annotations = {}
        for folder in tqdm(self.datasets[subset]):
            filenames = sorted(os.listdir(os.path.join(self.data_dir, folder)))
            label_files = [x for x in filenames if '.pts' in x]
            image_files = [x for x in filenames if '.pts' not in x]
            assert len(image_files) == len(label_files)

            for image_name, label_name in zip(image_files, label_files):
                assert os.path.splitext(image_name)[0] == os.path.splitext(label_name)[0]

                image_crop_name = folder.replace('/', '_') + '_' + image_name.replace(' ', '')
                image_crop_path = os.path.join(self.data_dir_img, subset, image_crop_name)
                image_crop_path_short = os.path.join('images', subset, image_crop_name)

                image_crop, annotation = self.process_300W(folder, image_name, label_name)
                cv2.imwrite(image_crop_path, image_crop)
                annotations[image_crop_path_short] = annotation

        with open(os.path.join(self.data_dir, f'{subset}.txt'), 'w') as f:
            for image_crop_path_short, annotation in annotations.items():
                f.write(image_crop_path_short + ' ')
                for x, y in annotation:
                    f.write(str(x) + ' ' + str(y) + ' ')
                f.write('\n')

    def split_data(self):
        """ Split 'test-dataset' into 'common' and 'challenge' test-datasets """
        with open(os.path.join(self.data_dir, 'test.txt'), 'r') as f:
            annotations = f.readlines()

        with open(os.path.join(self.data_dir, 'test_common.txt'), 'w') as f:
            for annotation in annotations:
                if 'ibug' not in annotation:
                    f.write(annotation)
        
        with open(os.path.join(self.data_dir, 'test_challenge.txt'), 'w') as f:
            for annotation in annotations:
                if 'ibug' in annotation:
                    f.write(annotation)

    def calculate_mean_lmk(self):
        """ Calculate mean-lmk from 'train' dataset """
        with open(os.path.join(self.data_dir, 'train.txt'), 'r') as f:
            annotations = f.readlines()
        
        annotations = [x.strip().split()[1:] for x in annotations]
        annotations = [[float(x) for x in anno] for anno in annotations]
        annotations = np.array(annotations)

        mean_lmk = np.mean(annotations, axis=0).tolist()
        mean_lmk = [str(x) for x in mean_lmk]

        with open(os.path.join(self.data_dir, 'mean-lmk.txt'), 'w') as f:
            f.write(' '.join(mean_lmk))

    def calculate_combined_mean_lmk(self):
        """ Calculate mean-lmk from 'train' dataset """
        with open(os.path.join(self.data_dir, 'train_300W-WFLW-lmk12.txt'), 'r') as f:
            annotations_1 = f.readlines()
        
        with open(os.path.join(self.data_dir, 'train_42dot.txt'), 'r') as f:
            annotations_2 = f.readlines()
        
        # Combine train data
        annotations = annotations_1 + annotations_2
        with open(os.path.join(self.data_dir, 'train_300W-WFLW-lmk12-42dot.txt'), 'w') as f:
            f.writelines(annotations)
        
        # Calculate combined mean-lmk
        annotations = [x.strip().split()[1:] for x in annotations]
        annotations = [[float(x) for x in anno] for anno in annotations]
        annotations = np.array(annotations)

        mean_lmk = np.mean(annotations, axis=0).tolist()
        mean_lmk = [str(x) for x in mean_lmk]
        
        with open(os.path.join(self.data_dir, 'mean-lmk_300W-WFLW-lmk12-42dot.txt'), 'w') as f:
            f.write(' '.join(mean_lmk))
        
    def clean_lmk(self):
        """ Leave only eye landmarks for Drowsiness """
        with open(os.path.join(self.data_dir, 'train_300W-WFLW.txt'), 'r') as f:
            annotations = f.readlines()
        
        annotations = [x.strip().split() for x in annotations]
        
        # Reduce
        reduced_annotations = []
        for x in annotations:
            path = x[0:1]
            landmarks = x[36*2+1:48*2+1]
            reduced_annotations.append(path + landmarks)

        with open(os.path.join(self.data_dir, 'train_300W-WFLW-lmk12.txt'), 'w') as f:
            for anno in reduced_annotations:
                f.write(f"{' '.join(anno)}\n")
        
    def run(self):
        self.process_dataset('train')
        self.process_dataset('test')
        self.split_data()
        self.calculate_mean_lmk()
        

if __name__ == "__main__":
    generator = DataGenerator(data_dir="datasets/300W-WFLW/train.txt", target_size=256)
    generator.clean_lmk()
    generator.calculate_combined_mean_lmk()
