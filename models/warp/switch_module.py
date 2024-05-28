from .rotated_modules import warped_modules, WaRPModule
import torch.nn as nn

def switch_module(module):
    new_children = {}

    for name, submodule in module.named_children(): # 遍历所有子模块
        if isinstance(submodule, nn.Conv2d) and hasattr(submodule, 'is_warp_conv'):

            switched = warped_modules[type(submodule)](submodule) # 把对应的模块替换为warp模块
            new_children[name] = switched # 按模块名称将替换后的warp模块存到字典中

        switch_module(submodule) # 递归调用，处理模块的子模块

    for name, switched in new_children.items(): # 遍历所有替换过的模块，将该模块替换原来的模块
        setattr(module, name, switched)

    return module.cuda()

