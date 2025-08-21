import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import torch
from PIL import Image
import glob
import imageio

def get_random_color():
    result = []
    for i in range(3):
        r = random.uniform(0,1)
        result.append(r)
    return tuple(result)

def plot_loss(ax ,train, valid, name):
    #train *=3
    #valid *=3
    func = np.argmax if 'loss' not in name else np.argmin
    idx = func(valid)
    value = valid[idx]
    t_color, v_color = get_random_color(), get_random_color()
    ran = np.arange(len(train))
    ax.plot(ran,train, label='train',color=t_color)
    ax.plot(ran,valid, label='valid',color=v_color)
    ax.scatter([idx],[value],color=v_color,label=f'opt value {value:.3f}')
    ax.set_title(name)
    ax.legend(loc='lower left')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss' if 'loss' in name else 'acc')
    return value

def plot(paths, name, col=4 ):
    num_rows = len(paths)//col
    num_rows += 1 if len(paths)%col > 0. else 0

    fig, axes = plt.subplots(num_rows, col, figsize=(12,8),dpi=800)
    func = np.argmin if 'loss' in name else np.argmax
    results = []


    for r in range(num_rows):
        for c in range(num_rows):
            if r*num_rows + c == len(paths) :
                break
            ax = axes[r*num_rows+c]
            path = paths[num_rows*r*c]
            with open(path,'rb') as f:
                log = pickle.load(f)
            best = plot_loss(ax,log[f'train_{name}'], log[f'valid_{name}'],name)
            results.append(best)


    idx = func(results)
    value = results[idx]
    fig.text(0.5,0.01,f'best experiment : {idx} best metric : {value:.4f}')
    plt.tight_layout(rect=[0,0.03,1,1])
    plt.savefig(f'logs/{name}.png')
    plt.show()


def get_R(loss,ada_loss):
    crr_mat = np.corrcoef(loss,ada_loss)
    R = crr_mat[0,1]
    return R

def plot_mem():
    with open('logs/mem_log100.pkl','rb') as f :
        log = pickle.load(f)
    log = np.array(log)
    plt.xlabel('epoch')
    plt.ylabel('GB')
    plt.title('memory usage')
    plt.plot(log[:,0],label='avail')
    plt.plot(log[:,1],label='used')
    plt.legend()
    plt.savefig('memlog.png')
    plt.show()

def plot_rl(path, name):
    with open(path,'rb') as f:
        log = pickle.load(f)
    reward = torch.tensor(log['reward']).cpu().numpy()
    acc = torch.tensor(log['acc']).cpu().numpy()

    plt.figure(figsize=(12,6),dpi=800)
    plt.plot(reward)
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.title('reward')
    plt.legend(loc='lower left')
    plt.savefig(f'{name}_reward.png')
    plt.show()

    plt.figure(figsize=(12,6),dpi=800)
    plt.plot(acc)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('accuracy')
    idx = np.argmax(acc)
    value = acc[idx]
    plt.scatter(idx,value, label=f'max : {value:.4f}')

    R = get_R(np.array(reward),np.array(acc))
    plt.figtext(0.5,0.01, f'correlation coefficient : {R:.4f} ')
    plt.legend(loc='lower left')
    plt.savefig(f'{name}_accuracy.png')
    plt.show()


def plot_angle_gif(path, duration=6, fps=30):
    image_files = sorted([
        os.path.join(path, f) 
        for f in os.listdir(path) 
        if f.lower().endswith(('png', 'jpg', 'jpeg'))
    ])
    total_frames = fps * duration
    num_images = len(image_files)

    # 샘플링: 이미지가 많으면 프레임 수에 맞춰 골고루 샘플링
    if num_images > total_frames:
        step = num_images / total_frames
        sampled_indices = [int(step * i) for i in range(total_frames)]
    else:
        sampled_indices = list(range(num_images))
    gif_path = os.path.join(path, 'angle_animation.gif')
    # 이미지 읽기
    images = [imageio.imread(image_files[idx]) for idx in sampled_indices]

    # gif 저장 - duration은 한 프레임당 표시 시간(초)
    imageio.mimsave(gif_path, images, fps=fps)

    print(f"GIF saved with {len(images)} frames, fps: {fps}, duration: {duration}s")