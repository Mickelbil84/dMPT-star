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
from utils import create_output_dir

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

def cal_performance(predVals, anchorPoints, trueLabels, lengths):
    '''
    Return the loss and number of correct predictions.
    :param predVals: the output of the final linear layer.
    :param anchorPoints: The anchor points of interest
    :param trueLabels: The expected clas of the corresponding anchor points.
    :param lengths: The legths of each of sequence in the batch
    :returns (loss, n_correct): The loss of the model and number of avg predictions.
    '''
    n_correct = 0
    total_loss = 0
    for predVal, anchorPoint, trueLabel, length in zip(predVals, anchorPoints, trueLabels, lengths):
        predVal = predVal.index_select(0, anchorPoint[:length])
        trueLabel = trueLabel[:length]
        loss = F.cross_entropy(predVal, trueLabel)
        total_loss += loss
        classPred = predVal.max(1)[1]
        n_correct += classPred.eq(trueLabel[:length]).sum().item()/length
    return total_loss, n_correct

def visualize_epoch(model):
    """
    Output vizualization images to visdom
    """
    MAP_IDX = 1
    encoder_map = valid_dataset[MAP_IDX]['map']
    scene_map = encoder_map[0,:,:]
    locations_map = torch.abs(encoder_map[1,:,:])

    # show model predictions
    s = encoder_map.size()
    with torch.no_grad():
        pred_map = model(encoder_map.view(1, s[0], s[1], s[2]).float().cuda()).cpu()
    pred_map = pred_map.numpy().reshape((24, 24, 2))
    
    pos_pred_map = pred_map[:, :, 0].reshape((24, 24))
    neg_pred_map = pred_map[:, :, 1].reshape((24, 24))

    pos_pred_map = cv2.resize(pos_pred_map, (480, 480))
    neg_pred_map = cv2.resize(neg_pred_map, (480, 480))

    vis.image(scene_map.numpy(), win='Scene', opts=dict(title="Scene"))
    vis.image(locations_map.numpy(), win='Locations', opts=dict(title="Locations"))
    vis.image(pos_pred_map, win='PosPred', opts=dict(title="PosPred"))
    vis.image(neg_pred_map, win='NegPred', opts=dict(title="NegPred"))

def train_epoch(model, trainingData, optimizer, epoch, train_losses, train_accuracy):
    '''
    Train the model for 1-epoch with data from wds
    '''
    model.train()
    total_loss = 0
    total_n_correct = 0

    train_enum = tqdm(trainingData, mininterval=2, desc='Train epoch %d' % epoch)

    # Train for a single epoch.
    for batch in train_enum:
        
        optimizer.zero_grad()
        encoder_input = batch['map'].float().cuda()
        predVal = model(encoder_input)

        # Calculate the cross-entropy loss
        loss, n_correct = cal_performance(
            predVal, batch['anchor'].cuda(), 
            batch['labels'].cuda(), 
            batch['length'].cuda()
        )
        loss.backward()
        optimizer.step_and_update_lr()
        total_loss += loss.item()
        total_n_correct += n_correct
        
        train_enum.set_description('Train (loss %.2f) epoch %d' % (loss.item(), epoch))

    total_loss = float(total_loss / len(train_dataset))
    accuracy = float(total_n_correct / len(train_dataset))
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
    
    logging.info('====> Epoch {}: Train set loss: {:.4f}, accuracy: {:.4f}'.format(epoch, total_loss, accuracy))


def eval_epoch(model, validationData, epoch, valid_losses, valid_accuracy):
    '''
    Evaluation for a single epoch.
        :param model: The Transformer Model to be trained.
    :param validataionData: The set of validation data.
    '''
    model.eval()
    total_loss = 0.0
    total_n_correct = 0.0

    valid_enum = tqdm(validationData, mininterval=2, desc='Train epoch %d' % epoch)

    with torch.no_grad():
        for batch in valid_enum:

            encoder_input = batch['map'].float().cuda()
            predVal = model(encoder_input)

            loss, n_correct = cal_performance(
                predVal, 
                batch['anchor'].cuda(), 
                batch['labels'].cuda(),
                batch['length'].cuda()
            )

            total_loss += loss.item()
            total_n_correct += n_correct

            valid_enum.set_description('Valid (loss %.2f) epoch %d' % (loss.item(), epoch))

    total_loss = float(total_loss / len(valid_dataset))
    accuracy = float(total_n_correct / len(valid_dataset))
    valid_losses.append(total_loss)
    valid_accuracy.append(accuracy)

    if True:
        vis.line(
            Y=np.asarray(valid_losses), 
            X=torch.arange(1, 1+len(valid_losses)),
            opts=dict(title="Loss"),
            win='Loss ' + args.expName,
            name='valid',
            update=True)

        vis.line(
            Y=np.asarray(valid_accuracy), 
            X=torch.arange(1, 1+len(valid_accuracy)),
            opts=dict(title="Accuracy"),
            win='Accuracy ' + args.expName,
            name='valid',
            update=True)
    
    logging.info('====> Epoch {}: Validation set loss: {:.4f}, accuracy: {:.4f}'.format(epoch, total_loss, accuracy))


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
    eval_losses = []
    train_accuracy = []
    eval_accuracy = []
    
    # Start training
    for epoch in range(start_epoch, start_epoch + args.epochs):
        visualize_epoch(model)
        train_epoch(model, train_loader, optimizer, epoch, train_losses, train_accuracy)
        eval_epoch(model, valid_loader, epoch, eval_losses, eval_accuracy)


if __name__ == '__main__':
    main()