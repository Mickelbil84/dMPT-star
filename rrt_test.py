import os
import random

import cv2
import tqdm
import time
import torch
import numpy as np
import networkx as nx

from unet.Models import UNet
from data import PathDataLoader
from utils import neural_render_path

EPS = 0.0001
SIGMA = 50

def model_sampling(img, s_x, s_y, t_x, t_y, paths_rendered):
    h, w = paths_rendered.shape
    coin_flip = (random.random() < 0.95)
    while True:
        x, y = random.randint(0, w-1), random.randint(0, h-1)
        if not coin_flip:
            return x, y
        if paths_rendered[y,x] > EPS:
            break
    x_ = max(min(int(np.random.normal(x, SIGMA)), 0), 479)
    y_ = max(min(int(np.random.normal(y, SIGMA)), 0), 479)
    return x_, y_
    

def uniform_sampling(img, s_x, s_y, t_x, t_y, paths_rendered):
    h, w, _ = img.shape
    return random.randint(0, w-1), random.randint(0, h-1)

def is_point_valid(img, x, y):
    return img[y,x,0] > EPS

def is_edge_valid(img, x0, y0, x1, y1):
    n = 100
    for i in range(n+1):
        t = i / n
        x = int(t * x1 + (1-t) * x0)
        y = int(t * y1 + (1-t) * y0)

        if img[y,x,0] < EPS:
            return False
    
    return True

def k_nearest_neighbors(points, x, y, k):
    return sorted(points, key=lambda p: (p[0]-x)**2 + (p[1]-y)**2)[:k]

def nearest(points, x, y):
    d = None
    x_near, y_near = None, None
    for x_, y_ in points:
        d_ = (x - x_)**2 + (y - y_)**2
        if d is None or d_ < d:
            d = d_
            x_near, y_near = x_, y_
    return x_near, y_near

def near_in_radius(points, x, y, r):
    res = []
    for x_, y_ in points:
        if (x-x_)**2 + (y-y_)**2 < r**2:
            res.append((x_, y_))
    return res

def steer(x0, y0, x1, y1, eta):
    d = dist(x0, y0, x1, y1)
    if d == 0:
        return None
    t = min(eta / d, 1)
    x_ = int(x1 * t + x0 * (1-t))
    y_ = int(y1 * t + y0 * (1-t))
    return x_, y_


def draw_graph(img, graph):
    for node in graph.nodes:
        x, y = node
        img = cv2.circle(img, (x,y), 2, (255,0,0), -1)
    for edge in graph.edges:
        x, y = edge[0]
        x_, y_ = edge[1]
        img = cv2.line(img, (x,y), (x_, y_), (255,0,0))
    return img

def dist(x0, y0, x1, y1):
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5


def build_rrt(img, s_x, s_y, t_x, t_y, paths_rendered, sampling_method, eta=5, max_points=1000):
    rrt = nx.Graph()
    rrt.add_node((s_x, s_y))
    cost = {}
    cost[(s_x,s_y)] = 0
    sampled_points = []
    success = False

    while True:
        p_rand = sampling_method(img, s_x, s_y, t_x, t_y, paths_rendered)
        if not is_point_valid(img, *p_rand):
            continue
        p_near = nearest(rrt.nodes, *p_rand)
        p_new = steer(*p_near, *p_rand, eta)

        sampled_points.append(p_rand)
        if len(sampled_points) > max_points:
            break

        if p_new is None or not is_edge_valid(img, *p_near, *p_new):
            continue

        neighbors = near_in_radius(rrt.nodes, *p_new, r=eta)
        rrt.add_node(p_new)
        p_min = p_near
        c_min = cost[p_near] + dist(*p_new, *p_near)

        for p_near in neighbors:
            if is_edge_valid(img, *p_near, *p_new):
                if cost[p_near] + dist(*p_new, *p_near) < c_min:
                    p_min = p_near
                    c_min = cost[p_near] + dist(*p_new, *p_near)
        
        rrt.add_edge(p_min, p_new, weight=dist(*p_min, *p_new))
        cost[p_new] = c_min

        for p_near in neighbors:
            if is_edge_valid(img, *p_new, *p_near):
                if cost[p_new] + dist(*p_new, *p_near) < cost[p_near]:
                    parents = list(rrt.neighbors(p_near))
                    for p_parent in parents:
                        rrt.remove_edge(p_parent, p_near)
                    rrt.add_edge(p_new, p_near, weight=dist(*p_near, *p_new))

        # Try connecting to the target
        p_connect = k_nearest_neighbors(rrt.nodes, t_x, t_y, k=15)
        for p in p_connect:
            if is_edge_valid(img, *p, t_x, t_y):
                rrt.add_edge(p, (t_x, t_y), weight=dist(*p, t_x, t_y))
                success = True
                break
        if success:
            break
        
        
    # For debug view, also add the sparse points we sampled
    for point in sampled_points:
        rrt.add_node(point)
    
    return rrt, success
    
def extract_data_sample(data_sample):
    img = data_sample["map"][0, :, :].numpy() * 255
    img = img.reshape(480, 480, 1)
    img = np.repeat(img, 3, 2)
    inpt = data_sample["map"].cpu().view(1, 2, 480, 480).float()

    s_x, s_y = data_sample["start"][1], data_sample["start"][0]
    t_x, t_y = data_sample["goal"][1], data_sample["goal"][0]
    start_point = data_sample["start"].float().cpu().view(-1, 1, 2)
    goal_point = data_sample["goal"].float().cpu().view(-1, 1, 2)

    s_x = int((s_x + 1) * 479 / 2)
    s_y = int((s_y + 1) * 479 / 2)
    t_x = int((t_x + 1) * 479 / 2)
    t_y = int((t_y + 1) * 479 / 2)

    return img, s_x, s_y, t_x, t_y, inpt, start_point, goal_point

def validate_scene(img, s_x, s_y, t_x, t_y):
    if not is_point_valid(img, s_x, s_y):
        return False
    if not is_point_valid(img, t_x, t_y):
        return False
    return True


if __name__ == "__main__":
    valid_dataset = PathDataLoader(
        env_list=list(range(500)),
        dataFolder=os.path.join("data", "forest", "val"),
    )
    test_dataset = PathDataLoader(
        env_list=list(range(1750, 1750+500)),
        dataFolder=os.path.join("data", "forest", "val"),
    )

    ##################
    # Load models
    ##################
    # model_lambd_0_1 = UNet(2,1, 20).cuda()
    # model_lambd_0_1.load_state_dict(torch.load("checkpoints/forest_lambd_0.1.pkl", map_location="cuda"))
    # model_lambd_0_01 = UNet(2,1, 20).cuda()
    # model_lambd_0_01.load_state_dict(torch.load("checkpoints/forest_lambd_0.01.pkl", map_location="cuda"))
    # model_lambd_0_001 = UNet(2,1, 20).cuda()
    # model_lambd_0_001.load_state_dict(torch.load("checkpoints/forest_lambd_0.001.pkl", map_location="cuda"))
    # model_lambd_0 = UNet(2,1, 20).cuda()
    # model_lambd_0.load_state_dict(torch.load("checkpoints/forest_lambd_0.pkl", map_location="cuda"))
    # model_lambd_1 = UNet(2,1, 20).cuda()
    # model_lambd_1.load_state_dict(torch.load("checkpoints/forest_lambd_1.pkl", map_location="cuda"))
    # model_n_10 = UNet(2,1, 10).cuda()
    # model_n_10.load_state_dict(torch.load("checkpoints/forest_n_10.pkl", map_location="cuda"))
    # model_n_20 = UNet(2,1, 20).cuda()
    # model_n_20.load_state_dict(torch.load("checkpoints/forest_n_20.pkl", map_location="cuda"))
    # model_n_50 = UNet(2,1, 50).cuda()
    # model_n_50.load_state_dict(torch.load("checkpoints/forest_n_50.pkl", map_location="cuda"))
    # models = {
    #     'lambd = 0.1': model_lambd_0_1,
    #     'lambd = 0.01': model_lambd_0_01,
    #     'lambd = 0.001': model_lambd_0_001,
    #     'lambd = 0': model_lambd_0,
    #     'lambd = 1': model_lambd_1,
    #     'n = 10': model_n_10,
    #     'n = 20': model_n_20,
    #     'n = 50': model_n_50,
    # }
    # model_width = UNet(2,1, 10).cuda()
    # model_width.load_state_dict(torch.load("checkpoints/forest_width_0.04.pkl", map_location="cuda"))
    # models = {
    #     'width = 0.04': model_width
    # }
    models = {}

    # Run tests on validation set
    cnt = 0
    success_rates = {model_name: 0 for model_name in models}
    vertices = {model_name: 0 for model_name in models}
    avg_time = {model_name: 0 for model_name in models}
    
    # Add baselines to results
    success_rates['RRT*'] = 0
    vertices['RRT*'] = 0
    avg_time['RRT*'] = 0
    success_rates['line'] = 0
    vertices['line'] = 0
    avg_time['line'] = 0
    for data_sample in tqdm.tqdm(valid_dataset):
        if data_sample is None:
            continue
        img, s_x, s_y, t_x, t_y, inpt, start_point, goal_point = extract_data_sample(data_sample)
        if not validate_scene(img, s_x, s_y, t_x, t_y):
            continue
        cnt += 1

        # Run baseline
        t0 = time.time()
        rrt, success = build_rrt(img, s_x, s_y, t_x, t_y, None, uniform_sampling, max_points=5000)
        t1 = time.time()
        success_rates['RRT*'] += 1 if success else 0
        vertices['RRT*'] += len(rrt.nodes) if success else 0
        avg_time['RRT*'] += (t1 - t0) if success else 0

        paths = [start_point.cpu()]
        for i in range(13):
            t = i / 12
            paths.append(t * goal_point + (1-t) * start_point)
        paths.append(goal_point.cpu())
        paths = torch.cat(
            paths, dim=1
        )
        paths_rendered = neural_render_path(paths, 480, paths.shape[1]).detach().cpu().view(480,480).numpy()
        t0 = time.time()
        rrt, success = build_rrt(img, s_x, s_y, t_x, t_y, paths_rendered, model_sampling, max_points=5000)
        t1 = time.time()
        success_rates['line'] += 1 if success else 0
        vertices['line'] += len(rrt.nodes) if success else 0
        avg_time['line'] += (t1 - t0) if success else 0

        for model_name in models:
            model = models[model_name]
            paths = model(inpt)
            paths = torch.cat(
                (start_point, paths, goal_point), dim=1
            )
            paths_rendered = neural_render_path(paths, 480, paths.shape[1]).detach().cpu().view(480,480).numpy()
            t0 = time.time()
            rrt, success = build_rrt(img, s_x, s_y, t_x, t_y, paths_rendered, model_sampling, max_points=5000)
            t1 = time.time()
            success_rates[model_name] += 1 if success else 0
            vertices[model_name] += len(rrt.nodes) if success else 0
            avg_time[model_name] += (t1 - t0) if success else 0

    
    print(cnt)
    print(success_rates)
    print(vertices)
    print(avg_time)



    # img, s_x, s_y, t_x, t_y, inpt = extract_data_sample(valid_dataset[0])
    # rrt, success = build_rrt(img, s_x, s_y, t_x, t_y, inpt, lambda img, s_x, s_y, t_x, t_y, inpt: model_sampling(img, s_x, s_y, t_x, t_y, inpt, model))
    # new_img = draw_graph(img, rrt)
    # cv2.imwrite('test.png', new_img)
    # print(success)

    # cv2.imwrite("test.png", img)


