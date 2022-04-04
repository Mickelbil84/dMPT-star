import sys
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import visdom
from torch.utils.data import DataLoader
from tqdm import tqdm
from vit_pytorch import ViT

from data import PaddedSequence, PathDataLoader
from utils import create_output_dir, neural_render_path

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
parser.add_argument("--batch-size", type=int, default=64, help="batch size")
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
    "--n-points", type=int, default=20, help="number of points to predict per path"
)
parser.add_argument(
    "--img-size", type=int, default=480, help="image size of input and rendered paths"
)


#######
# Init
#######

args = parser.parse_args()
args.data_folder = os.path.join(args.data_folder, args.expName)
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


def visualize_epoch(model, print_paths=True):
    """
    Output visualization images to visdom
    """
    MAP_IDX = 5
    model.eval()
    with torch.no_grad():
        for i in range(MAP_IDX):
            encoder_map = valid_dataset[i]["map"]
            scene_map = encoder_map[0, :, :]
            # locations_map = torch.abs(encoder_map[1, :, :])

            # show model predictions
            s = encoder_map.size()
            black_channel = torch.zeros(1, 1, s[1], s[1])
            x = torch.cat((encoder_map.view(1, s[0], s[1], s[2]), black_channel), dim=1)
            paths = torch.tanh(model(x.float().cuda()).view(-1, args.n_points, 2))
            paths = torch.cat(
                (
                    valid_dataset[i]["start"].float().cuda().view(-1, 1, 2),
                    paths,
                    valid_dataset[i]["goal"].float().cuda().view(-1, 1, 2),
                ),
                dim=1,
            )

            paths_rendered = neural_render_path(paths, args.img_size, args.n_points + 2)
            # endpoints_rendered = neural_render_endpoints(paths, args.img_size, args.n_points)
            # cspace_rendered = (
            #     encoder_map[0, :, :].view(-1, args.img_size, args.img_size).cuda()
            # )

            if print_paths:
                print(paths.detach().cpu())

            # enc_output, *_ = model.encoder(encoder_map.view(1, s[0], s[1], s[2]).float().cuda())
            # print(enc_output)

            # masked_paths = paths_rendered * (1 - cspace_rendered)

            paths_map = paths_rendered.detach().cpu().numpy()[0]
            # vis.image(image, win='Paths', opts=dict(title="Paths"))
            # endpoints = torch.abs(endpoints_rendered.detach().cpu()).numpy()[0]
            # vis.image(endpoints, win='Endpoints', opts=dict(title="Endpoints"))

            # vis.image(scene_map.numpy(), win='Scene', opts=dict(title="Scene"))
            # vis.image(locations_map.numpy(), win='Locations', opts=dict(title="Locations"))

            # Draw the scene with the paths
            scene = (scene_map.numpy() + paths_map) / 2.0
            vis.image(scene, win="Scene_" + str(i), opts=dict(title="Scene_" + str(i)))


def equal_length_loss(paths):
    n_points = paths.shape[1]

    loss = torch.tensor(0.0).cuda()
    for i in range(n_points - 1):
        tmp = F.mse_loss(paths[:, i, :], paths[:, i + 1, :])
        # if i == 0 or i == n_points - 2:
        #     loss += tmp
        # else:
        #     loss -= tmp
        loss += tmp
    return loss


def train_epoch(model, trainingData, optimizer, epoch, train_losses, train_accuracy):
    """
    Train the model for 1-epoch with data from wds
    """
    model.train()
    cnt = 0
    total_loss = 0
    total_n_correct = 0

    train_enum = tqdm(trainingData, mininterval=2, desc="Train epoch %d" % epoch)

    # Train for a single epoch.
    for batch in train_enum:

        optimizer.zero_grad()
        model.zero_grad()

        # Fetch training data
        encoder_input = batch["map"].float().cuda()
        # cspace_rendered = 1.0 - encoder_input[:, 0,:,:].view(-1, args.img_size, args.img_size) # TODO: Apply blurring maybe?
        cspace_rendered = encoder_input[:, 0, :, :].view(
            -1, args.img_size, args.img_size
        )
        start_point = batch["start"].float().cuda().view(-1, 1, 2)
        goal_point = batch["goal"].float().cuda().view(-1, 1, 2)
        batch_size = encoder_input.shape[0]

        black_channel = torch.zeros(batch_size, 1, encoder_input.shape[2], encoder_input.shape[3]).cuda()
        x = torch.cat((encoder_input, black_channel), dim=1)

        # Predict path given the input
        paths = torch.tanh(model(x).view(-1, args.n_points, 2))
        paths = torch.cat(
            (start_point, paths, goal_point), dim=1
        )  # Force source and target by appending them to the start and end of the rendered path

        # Render the path using a neural renderer and compute unsupervised loss
        paths_rendered = neural_render_path(paths, args.img_size, args.n_points + 2)

        zeros_image = (
            torch.zeros((batch_size, args.img_size * args.img_size)).float().cuda()
        )
        masked_paths = paths_rendered * (1 - cspace_rendered)

        # loss = torch.sum(masked_paths * masked_paths) / batch_size

        # loss = equal_length_loss(paths)
        loss = F.mse_loss(
            masked_paths.view(-1, args.img_size * args.img_size), zeros_image
        )
        loss += 0.001 * equal_length_loss(paths)
        # loss = F.l1_loss(1.0 / (masked_paths.view(-1, args.img_size * args.img_size) + 1.0), zeros_image)
        # loss += F.l1_loss(paths_rendered.view(-1, args.img_size * args.img_size), zeros_image) * 1000
        # loss += F.mse_loss(start_point.view(-1, 2), paths[:, 0, :].view(-1,2))
        # loss += F.mse_loss(goal_point.view(-1, 2), paths[:, args.n_points-1, :].view(-1,2))
        loss.backward()
        optimizer.step()
        # optimizer.step_and_update_lr()

        cnt += 1
        total_loss += loss.item()
        n_correct = 0
        total_n_correct += n_correct

        train_enum.set_description("Train (loss %.6f) epoch %d" % (loss.item(), epoch))

        if cnt % 10 == 0:
            visualize_epoch(model, False)
            model.train()

    total_loss = float(total_loss / cnt)
    accuracy = float(total_n_correct / cnt)
    train_losses.append(total_loss)
    train_accuracy.append(accuracy)

    if True:
        vis.line(
            Y=np.asarray(train_losses),
            X=torch.arange(1, 1 + len(train_losses)),
            opts=dict(title="Loss"),
            win="Loss " + args.expName,
            name="train",
        )

        # vis.line(
        #     Y=np.asarray(train_accuracy),
        #     X=torch.arange(1, 1+len(train_accuracy)),
        #     opts=dict(title="Accuracy"),
        #     win='Accuracy ' + args.expName,
        #     name='train')

    logging.info(
        "====> Epoch {}: Train set loss: {:.6f}, accuracy: {:.6f}".format(
            epoch, total_loss, accuracy
        )
    )


def main():
    start_epoch = 1

    model = ViT(
        image_size=480,
        patch_size=32,
        num_classes=args.n_points * 2,
        dim=512,
        depth=3,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    ).cuda()

    if args.checkpoint != "":
        checkpoint_args_path = os.path.dirname(args.checkpoint) + "/args.pth"
        checkpoint_args = torch.load(checkpoint_args_path)
        start_epoch = checkpoint_args[3]
        model.load_state_dict(torch.load(args.checkpoint))

    optimizer = optim.Adam(model.parameters(), lr=1e-7)

    # Keep track of losses
    train_losses = []
    train_accuracy = []

    # Start training
    for epoch in range(start_epoch, start_epoch + args.epochs):
        visualize_epoch(model)
        train_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            train_losses,
            train_accuracy,
        )
        torch.save(model.state_dict(), "checkpoints/model_vit.pkl")


if __name__ == "__main__":
    main()
