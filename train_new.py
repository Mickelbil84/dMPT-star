import math
import os
import sys
import argparse

import cv2
import visdom
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange

from data import PathDataLoader, PaddedSequence
from transformer.models import Transformer
from transformer.optim import ScheduledOptim
from utils import create_output_dir, neural_render_path, nerual_render_endpoints

torch.cuda.empty_cache()

############
# Arguments
############

print(torch.__version__)

parser = argparse.ArgumentParser(description='PyTorch dMPT-*')

parser.add_argument('--expName', type=str, default='forest', metavar='E', help='Experiment name')
parser.add_argument('--gpu', default=0, metavar='G', type=int, help='GPU device ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--checkpoint', type=str, default='', metavar='C', help='checkpoint path')
parser.add_argument('--epochs', type=int, default=100, metavar='K', help='number of train epochs')

parser.add_argument('--env-list-train', type=int, default=1750, help='range of enviroments to collect train data from')
parser.add_argument('--env-list-val', type=int, default=1750, help='range of enviroments to collect validation data from')
parser.add_argument('--data-folder', type=str, default='data', metavar='D', help='root to data folder')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--train-workers', type=int, default=8, help='number of CPU threads for train data')
parser.add_argument('--val-workers', type=int, default=4, help='number of CPU threads for validation data')
parser.add_argument('--n-points', type=int, default=50, help='number of points to predict per path')


#######
# Init
#######

args = parser.parse_args()
args.data_folder = os.path.join(args.data_folder, args.expName)
args.expName = os.path.join('checkpoints', args.expName)
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if __name__ == "__main__":
    logging = create_output_dir(args)
    vis = visdom.Visdom(env=args.expName.replace('\\', '_'))


###############
# Prepare data
###############

if __name__ == "__main__":
    logging.info("Building dataset.")

    # Train data
    train_dataset = PathDataLoader(env_list=list(range(args.env_list_train)), dataFolder=os.path.join(args.data_folder, 'train'))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=PaddedSequence, num_workers=args.train_workers)

    # Validation data
    valid_dataset = PathDataLoader(env_list=list(range(args.env_list_val)), dataFolder=os.path.join(args.data_folder, 'val'))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=PaddedSequence, num_workers=args.val_workers)

    logging.info("Dataset ready!")


###############
# Training
###############

def visualize_epoch(model):
    """
    Output vizualization images to visdom
    """
    MAP_IDX = 10
    encoder_map = valid_dataset[MAP_IDX]['map']
    scene_map = encoder_map[0,:,:]
    locations_map = torch.abs(encoder_map[1,:,:])

    # show model predictions
    s = encoder_map.size()
    paths = model(encoder_map.view(1, s[0], s[1], s[2]).float().cuda())
    paths = torch.cat((
        valid_dataset[MAP_IDX]['start'].float().cuda().view(-1, 1, 2), 
        paths[:, 1:(args.n_points-1), :],
        valid_dataset[MAP_IDX]['goal'].float().cuda().view(-1, 1, 2)), dim=1)


    paths_rendered = neural_render_path(paths, 480, args.n_points)
    # endpoints_rendered = nerual_render_endpoints(paths, 480, args.n_points)

    print(paths.detach().cpu())
    
    paths_map = paths_rendered.detach().cpu().numpy()[0]
    # vis.image(image, win='Paths', opts=dict(title="Paths"))
    # endpoints = torch.abs(endpoints_rendered.detach().cpu()).numpy()[0]
    # vis.image(endpoints, win='Endpoints', opts=dict(title="Endpoints"))

    # vis.image(scene_map.numpy(), win='Scene', opts=dict(title="Scene"))
    vis.image(locations_map.numpy(), win='Locations', opts=dict(title="Locations"))

    # Draw the scene with the paths
    scene = (scene_map.numpy() + paths_map) / 2.0
    vis.image(scene, win='Scene', opts=dict(title="Scene"))


def location_loss(endpoints_rendered, locations_rendered):
    """
    Return the similarity between the start position and end position
    """
    factor = 1 / (math.sqrt(endpoints_rendered.shape[0]) * 480)
    loss = torch.norm(torch.relu(endpoints_rendered) - torch.relu(locations_rendered))
    loss += torch.norm(torch.relu(-endpoints_rendered) - torch.relu(-locations_rendered))
    return loss * factor


def train_epoch(model, trainingData, optimizer, epoch, train_losses, train_accuracy):
    '''
    Train the model for 1-epoch with data from wds
    '''
    model.train()
    cnt = 0
    total_loss = 0
    total_n_correct = 0

    train_enum = tqdm(trainingData, mininterval=2, desc='Train epoch %d' % epoch)

    criterion = torch.nn.MSELoss()

    # Train for a single epoch.
    for batch in train_enum:
        
        optimizer.zero_grad()
        
        encoder_input = batch['map'].float().cuda()
        paths = model(encoder_input)
        # endpoints_rendered = nerual_render_endpoints(paths, 480, args.n_points)

        # TODO: Apply blurring maybe?
        cspace_rendered = 1.0 - encoder_input[:, 0,:,:].view(-1, 480, 480)
        # locations_rendered = encoder_input[:,1,:,:].view(-1, 480, 480)
        # loss = 0 * torch.norm(paths_rendered * cspace_rendered) + location_loss(endpoints_rendered, locations_rendered)

        # Force source and target
        # loss = criterion(paths[:, 0, :].view(-1, 2), batch['start'].float().cuda()) + criterion(paths[:, args.n_points-1, :].view(-1, 2), batch['goal'].float().cuda())
        paths = torch.cat((
            batch['start'].float().cuda().view(-1, 1, 2), 
            paths[:, 1:(args.n_points-1), :],
            batch['goal'].float().cuda().view(-1, 1, 2)), dim=1)
        paths_rendered = neural_render_path(paths, 480, args.n_points)
        loss = torch.norm(paths_rendered * cspace_rendered) / (math.sqrt(paths_rendered.shape[0]) * 480)
        # loss += 0.001 * torch.norm(paths_rendered) / (math.sqrt(paths_rendered.shape[0]) * 480) # Also make sure the path is short

        n_correct = 0
        
        loss.backward()
        optimizer.step_and_update_lr()
        total_loss += loss.item()
        total_n_correct += n_correct
        cnt += 1
        
        train_enum.set_description('Train (loss %.6f) epoch %d' % (loss.item(), epoch))

    total_loss = float(total_loss / cnt)
    accuracy = float(total_n_correct / cnt)
    train_losses.append(total_loss)
    train_accuracy.append(accuracy)

    if True:
        vis.line(
            Y=np.asarray(train_losses), 
            X=torch.arange(1, 1+len(train_losses)),
            opts=dict(title='Loss'),
            win='Loss ' + args.expName,
            name='train')

        vis.line(
            Y=np.asarray(train_accuracy), 
            X=torch.arange(1, 1+len(train_accuracy)),
            opts=dict(title="Accuracy"),
            win='Accuracy ' + args.expName,
            name='train')
    
    logging.info('====> Epoch {}: Train set loss: {:.6f}, accuracy: {:.6f}'.format(epoch, total_loss, accuracy))

def main():
    start_epoch = 1

    model_args = dict(
        n_layers=6, 
        n_heads=3, 
        d_k=512, 
        d_v=256, 
        d_model=512, 
        d_inner=1024, 
        pad_idx=None,
        n_position=40*40, 
        dropout=0.1,
        train_shape=[24, 24],
        predict_paths=True,
        n_points=args.n_points,
    )
    model = Transformer(**model_args).cuda()

    if args.checkpoint != '':
        checkpoint_args_path = os.path.dirname(args.checkpoint) + '/args.pth'
        checkpoint_args = torch.load(checkpoint_args_path)
        start_epoch = checkpoint_args[3]
        model.load_state_dict(torch.load(args.checkpoint))

    optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
        lr_mul=0.5,
        d_model=256,
        n_warmup_steps=3200
    )

    # Keep track of losses
    train_losses = []
    train_accuracy = []
    
    # Start training
    for epoch in range(start_epoch, start_epoch + args.epochs):
        visualize_epoch(model)
        train_epoch(model, train_loader, optimizer, epoch, train_losses, train_accuracy)


if __name__ == '__main__':
    main()