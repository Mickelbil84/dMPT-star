from __future__ import print_function

import os
import time
import logging
import subprocess
from datetime import timedelta

import numpy as np

import torch
from torch.autograd import Variable

import neural_renderer as nr

def nerual_render_endpoints(path, image_size, n_points, eps=0.05):
    """
    Get paths and return the image of start and end locations
    """
    batch_size = path.shape[0]

    start_position = path[:, 0].view(batch_size, 1, 2)
    end_position = path[:, n_points - 1].view(batch_size, 1, 2)

    epsilons = np.array([
        [-eps, -eps, 0.0],
        [-eps, eps, 0.0],
        [eps, eps, 0.0],
        [eps, -eps, 0.0]
    ])
    epsilons = torch.from_numpy(epsilons).float().cuda()

    zeros = torch.zeros((batch_size, 4, 1)).cuda()
    vertices_start = torch.cat((start_position, start_position, start_position, start_position), dim=1)
    vertices_start = torch.cat((vertices_start, zeros + 1.0), dim=2)
    vertices_start = vertices_start + epsilons

    vertices_end = torch.cat((end_position, end_position, end_position, end_position), dim=1)
    vertices_end = torch.cat((vertices_end, zeros + 1.0), dim=2)
    vertices_end = vertices_end + epsilons

    faces = torch.from_numpy(np.array([
        [0,1,2],
        [0,2,3]
    ])).int().cuda().view(-1, 2, 3)
    faces = faces.repeat(batch_size, 1, 1)

    renderer = nr.Renderer(image_size=image_size, camera_mode='look', perspective=False)
    images_start = renderer(vertices_start, faces, mode='silhouettes')
    images_end = renderer(vertices_end, faces, mode='silhouettes')
    return images_end - images_start


def neural_render_path(path, image_size, n_points, eps=0.01):
    """
    Get a path *tensor* and generate a differentiable image from it
    """
    batch_size = path.shape[0]

    # Generate 4n points, half with x-coord shifted by epsilon and half with y
    zeros = torch.zeros((batch_size, 2*n_points, 1)).cuda()
    vertices = torch.cat((path, path), dim=1)
    vertices = torch.cat((vertices, zeros + 1.0), dim=2)

    zeros = torch.zeros((batch_size, n_points, 1)).cuda()

    plus_eps_x = torch.cat((zeros+eps, zeros, zeros), dim=2)
    minus_eps_x = torch.cat((zeros-eps, zeros, zeros), dim=2)
    eps_x = torch.cat((minus_eps_x, plus_eps_x), dim=1)

    plus_eps_y = torch.cat((zeros, zeros+eps, zeros), dim=2)
    minus_eps_y = torch.cat((zeros, zeros-eps, zeros), dim=2)
    eps_y = torch.cat((minus_eps_y, plus_eps_y), dim=1)
    
    vertices = torch.cat((vertices+eps_x, vertices+eps_y), dim=1)
    vertices = vertices.float()
    
    # Generate the face buffer (which will always be constant)
    faces = []
    for i in range(n_points - 1):
        bl = i+1
        tl = i
        tr = i + n_points
        br = i+1 + n_points
        faces.append([bl, tl, tr])
        faces.append([bl, tr, br])
        bl = i+1 + 2 * n_points
        tl = i + 2 * n_points
        tr = i + 3 * n_points
        br = i+1 + 3 * n_points
        faces.append([bl, tl, tr])
        faces.append([bl, tr, br])
    faces = torch.from_numpy(np.array(faces)).int().cuda().view(-1, n_points*4 - 4, 3)
    faces = faces.repeat(batch_size, 1, 1)

    # Render the path
    # camera_distance = 0.7358397483825684 # Computed by binary search
    renderer = nr.Renderer(image_size=image_size, camera_mode='look', perspective=False)
    # renderer.eye = nr.get_points_from_angles(camera_distance, 0, 0)
    images = renderer(vertices, faces, mode='silhouettes')
    return images

##
# Code for the logging taken from the VoiceLoop repository
# Link: https://github.com/facebookarchive/loop
##
class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


##
# Code for the logging taken from the VoiceLoop repository
# Link: https://github.com/facebookarchive/loop
##
def create_output_dir(opt):
    filepath = os.path.join(opt.expName, 'main.log')

    if not os.path.exists(opt.expName):
        os.makedirs(opt.expName)

    # Safety check
    if os.path.exists(filepath) and opt.checkpoint == "":
        logging.warning("Experiment already exists!")

    # Create logger
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # quite down visdom
    logging.getLogger("requests").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    logger.info(opt)
    return logger

##
# Code taken from original MPT article
##
def geom2pix(pos, res=0.05, size=(480, 480)):
    """
    Convert geometrical position to pixel co-ordinates. The origin 
    is assumed to be at [image_size[0]-1, 0].
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :returns (int, int): The associated pixel co-ordinates.
    NOTE: The Pixel co-ordinates are represented as follows:
    (0,0)------ X ----------->|
    |                         |  
    |                         |  
    |                         |  
    |                         |  
    Y                         |
    |                         |
    |                         |  
    v                         |  
    ---------------------------  
    """
    return (np.int(np.floor(pos[0]/res)), np.int(size[0]-1-np.floor(pos[1]/res)))