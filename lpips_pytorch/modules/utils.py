from collections import OrderedDict
import os
import torch


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

"""
def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/lpips/weights/v{version}/{net_type}.pth'

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict

"""

def get_state_dict(net_type: str = 'alex', version: str = '0.1', local_path: str = './weights'):
    # 构建本地文件路径
    file_name = f'{net_type}.pth'
    file_path = '/public_bme2/bme-wangqian2/zhongach2024/PETMR-PETCT/lpips_pytorch/vgg.pth'

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Weight file not found at: {file_path}")

    # 从本地文件加载权重
    old_state_dict = torch.load(file_path,map_location=None if torch.cuda.is_available() else torch.device('cpu'))

    # 重命名字典中的键
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
