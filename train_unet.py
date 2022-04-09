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
    "--epochs", type=int, default=1000, metavar="K", help="number of train epochs"
)
parser.add_argument(
    "--lambd", type=float, default=0.1, help="lambda factor when combining the regularization loss"
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
    "--n-points", type=int, default=10, help="number of points to predict per path"
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
    vis = visdom.Visdom(env=args.expName.replace("\\", "_"))


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


def train_epoch(model, trainingData, optimizer, epoch, train_losses):
    """
    Train the model for one epoch
    """
    model.train()
    cnt = 0
    total_loss = 0
    grads = []

    train_enum = tqdm(trainingData, mininterval=2, desc="Train epoch %d" % epoch)

    # Train for a single epoch.
    for batch in train_enum:
        optimizer.zero_grad()
        model.zero_grad()

        # Get input data
        encoder_input = batch["map"].float().cuda()
        start_point = batch["start"].float().cuda().view(-1, 1, 2)
        goal_point = batch["goal"].float().cuda().view(-1, 1, 2)

        # Predict path given the input
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
        clr_loss = loss.item()
        loss += args.lambd * equal_length_loss(paths)
        loss.backward()
        optimizer.step()

        cnt += 1
        total_loss += clr_loss

        train_enum.set_description("Train (loss %.6f) epoch %d" % (loss.item(), epoch))

        #Check grad norm
        if args.show_grads:
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            grads.append(total_norm)
            vis.line(
                Y=np.asarray(grads), 
                opts=dict(title="Grad"),
                win="Gard " + args.expName,
                name="grad",
            )


    total_loss = float(total_loss / cnt)
    train_losses.append(total_loss)

    vis.line(
        Y=np.asarray(train_losses),
        X=torch.arange(1, 1 + len(train_losses)),
        opts=dict(title="Loss"),
        win="Loss " + args.expName,
        name="train",
    )

    logging.info(
        "====> Epoch {}: Train set loss: {:.6f}".format(
            epoch, total_loss
        )
    )


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
        loss += args.lambd * equal_length_loss(paths)
        
        cnt += 1
        total_loss += loss.item()

        valid_enum.set_description("Valid (loss %.6f) epoch %d" % (loss.item(), epoch))

    total_loss = float(total_loss / cnt)
    valid_losses.append(total_loss)

    vis.line(
        Y=np.asarray(valid_losses),
        X=torch.arange(1, 1 + len(valid_losses)),
        opts=dict(title="Loss"),
        win="Loss " + args.expName,
        name="valid",
        update=True
    )

    logging.info(
        "====> Epoch {}: Valid set loss: {:.6f}".format(
            epoch, total_loss
        )
    )



def main():

    model_unet = UNet(2, 1, n_points=args.n_points).cuda()
    optimizer_unet = optim.Adam(model_unet.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer_unet, step_size=4, gamma=0.1)

    # Keep track of losses
    train_losses = []
    valid_losses = []

    # Start training
    for epoch in range(1, 1 + args.epochs):
        visualize_model(model_unet, valid_dataset, vis, args.img_size)
        train_epoch(
            model_unet,
            train_loader,
            optimizer_unet,
            epoch,
            train_losses
        )
        valid_epoch(
            model_unet, valid_loader, epoch, valid_losses
        )
        scheduler.step()
        torch.save(model_unet.state_dict(), "{}.pkl".format(args.expName))


if __name__ == "__main__":
    main()
