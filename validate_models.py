import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import visdom
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PaddedSequence, PathDataLoader
from unet.Models import UNet
from utils import create_output_dir, neural_render_path, visualize_model

torch.cuda.empty_cache()

############
# Arguments
############

print(torch.__version__)

parser = argparse.ArgumentParser(description="PyTorch dMPT-*")

parser.add_argument(
    "--expName", type=str, default="forest", metavar="E", help="Experiment name"
)
parser.add_argument("--gpu", default=0, metavar="G", type=int, help="GPU device ID")
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--checkpoint", type=str, default="", metavar="C", help="checkpoint path"
)
parser.add_argument(
    "--epochs", type=int, default=1, metavar="K", help="number of train epochs"
)
parser.add_argument(
    "--lambd", type=float, default=0.001, help="lambda factor when combining the regularization loss"
)

parser.add_argument(
    "--env-list-train",
    type=int,
    default=1750,
    help="range of environments to collect train data from",
)
parser.add_argument(
    "--env-list-val",
    type=int,
    default=1750,
    help="range of environments to collect validation data from",
)
parser.add_argument(
    "--data-folder", type=str, default="data", metavar="D", help="root to data folder"
)
parser.add_argument("--batch-size", type=int, default=16, help="batch size")
parser.add_argument(
    "--train-workers", type=int, default=8, help="number of CPU threads for train data"
)
parser.add_argument(
    "--val-workers",
    type=int,
    default=4,
    help="number of CPU threads for validation data",
)
parser.add_argument(
    "--n-points", type=int, default=50, help="number of points to predict per path"
)
parser.add_argument(
    "--img-size", type=int, default=480, help="image size of input and rendered paths"
)
parser.add_argument(
    "--show-grads", type=bool, default=False, help="set true to plot grad's norm (for debug)"
)


#######
# Init
#######

args = parser.parse_args()
args.data_folder = os.path.join(args.data_folder, args.expName.split('_')[0])
args.expName = os.path.join("checkpoints", args.expName)
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if __name__ == "__main__":
    logging = create_output_dir(args)


###############
# Prepare data
###############

if __name__ == "__main__":
    logging.info("Building dataset.")

    # Train data
    train_dataset = PathDataLoader(
        env_list=list(range(args.env_list_train)),
        dataFolder=os.path.join(args.data_folder, "train"),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=PaddedSequence,
        num_workers=args.train_workers,
    )

    # Validation data
    valid_dataset = PathDataLoader(
        env_list=list(range(args.env_list_val)),
        dataFolder=os.path.join(args.data_folder, "val"),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        collate_fn=PaddedSequence,
        num_workers=args.val_workers,
    )

    logging.info("Dataset ready!")


###############
# Training
###############

def equal_length_loss(paths):
    """
    A regularization loss, resembling spring dynamics (or Nesterov's "Worst function in the world")
    Given a path (p0, ..., p_n), return:

    ||p_1-p0||^2 + ... + ||p_n - p_{n-1}||^2

    Where p_0 and p_n are fixed per scene
    """
    n_points = paths.shape[1]

    loss = torch.tensor(0.0).cuda()
    for i in range(n_points - 1):
        loss += F.mse_loss(paths[:, i, :], paths[:, i + 1, :])
    return loss


def clearance_loss(masked_paths):
    """
    Estimate the line integral of clearance of the path,
    by counting the pixels in the intersection of the path with the obstacles.

    We do that by returning just the squared norm of the masked paths matrix.
    """
    return F.mse_loss(masked_paths, torch.zeros_like(masked_paths).float().cuda())


def valid_epoch(model, validData, epoch, valid_losses):
    """
    Train the model for one epoch
    """
    model.eval()
    cnt = 0
    total_loss = 0

    valid_enum = tqdm(validData, mininterval=2, desc="Valid epoch %d" % epoch)
    
    for batch in valid_enum:

        # Get input data
        encoder_input = batch["map"].float().cuda()
        start_point = batch["start"].float().cuda().view(-1, 1, 2)
        goal_point = batch["goal"].float().cuda().view(-1, 1, 2)

        # Predict path given the input
        with torch.no_grad():
            paths = model(encoder_input)
        paths = torch.cat(
            (start_point, paths, goal_point), dim=1
        )  # Force source and target by appending them to the start and end of the rendered path

        # Render the path using a neural renderer and compute unsupervised loss
        # Also use the obstacle image and overlay them
        paths_rendered = neural_render_path(paths, args.img_size, args.n_points + 2)
        cspace_rendered = encoder_input[:, 0, :, :].view(
            -1, args.img_size, args.img_size
        )
        masked_paths = paths_rendered * (1 - cspace_rendered)

        # Compute the loss and optimize
        loss = clearance_loss(masked_paths.view(-1, args.img_size * args.img_size))

        cnt += 1
        total_loss += loss.item()

        valid_enum.set_description("Valid (loss %.6f) epoch %d" % (loss.item(), epoch))

    total_loss = float(total_loss / cnt)
    valid_losses.append(total_loss)

    logging.info(
        "====> Epoch {}: Valid set loss: {:.6f}".format(
            epoch, total_loss
        )
    )



def main():

    model_unet = UNet(2, 1, n_points=args.n_points).cuda()
    model_unet.load_state_dict(torch.load('checkpoints/forest_n_50.pkl', map_location='cuda'))

    # Keep track of losses
    valid_losses = []

    # Start training
    for epoch in range(1, 1 + args.epochs):
        valid_epoch(
            model_unet, valid_loader, epoch, valid_losses
        )

if __name__ == "__main__":
    main()
