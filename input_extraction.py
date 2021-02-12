"""
The code below are utilities for extracting and processing rendered images from the environment.
It uses the torchvision package, which makes it easy to compose image transforms.
Once you run the cell it will display an example patch that it extracted.
"""

import torchvision.transforms as T
from PIL import Image
import torch
import numpy as np

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # 小车中心位置


def get_screen(env, device):
    """
    Returned screen requested by gym is 400x600x3, but is sometimes larger such as 800x1200x3.
    """
    # 打开一个绘制窗口，绘制当前状态， model=rgb_array返回当前窗口的像素值
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))  # Transpose it into torch order (CHW-颜色、高度、宽度).
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape  # (3, 400, 600)
    # 小车大概在高40%（400X0.4=160）到 80%（400X0.8=320）之间
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


