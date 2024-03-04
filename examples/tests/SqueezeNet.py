import hecate as hc
import sys
import poly
from poly.models.SqueezeNet import *
from poly.MPCB import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image
import numpy as np
from random import *
import pprint


from pathlib import Path

seed(100)
source_path = Path(__file__).resolve()
source_dir = source_path.parent
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=str(source_dir)+"/../data/CIFAR10", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
])),
batch_size=128, shuffle=False,
num_workers=4, pin_memory=True)


# def roll(A, i) :
#     return A.rotate(-i)
def getModel():
    model = torch.nn.DataParallel(squeezenet())
    model_dict = torch.load(str(source_dir)+"/../data/squeezeNet_silu_avgpool_model", map_location=torch.device('cpu'))
    model.module.load_state_dict(model_dict)
    model = model.eval()
    return model


def preprocess(x):
    # print(x.shape)
    initial_shapes = {
    # Constant
    "nt" : 2**16,
    "bb" : 32,
    # Input Characteristics (Cascaded)
    "ko" : 1,
    "ho" : 32,
    "wo" : 32
    }
    conv1_shapes = CascadeConv(initial_shapes, model.module.conv_1.Conv2d)
    close = shapeClosure(**conv1_shapes)
    return close["MPP"](x)[0]

def process(x) : 
    model = getModel()
    torch_res = model(x) 
    torch_res = torch_res.cpu().detach().numpy()[0]
    return torch_res

def postprocess(res, torch_res) : 
    torch_res_size = 1
    for i in range(len(torch_res.shape)):
        torch_res_size *= torch_res.shape[i]
    return res[0,:torch_res_size].reshape(torch_res.shape) *32
    # return res[0,:torch_res_size].reshape(torch_res.shape)




if __name__ == "__main__" :

    from random import *
    import sys
    from pathlib import Path
    import time 
    from PIL import Image
    model = getModel()
    model = model.eval()

    a_compile_type = sys.argv[1]
    a_compile_opt = int(sys.argv[2])
    hc.setLibnHW(sys.argv)
    stem = Path(__file__).stem
    
    hevm = hc.HEVM()
    stem = Path(__file__).stem
    hevm.load (f"traced/_hecate_{stem}.cst", f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm")

    (input, target) = val_loader.dataset[0]
    input_var = input.unsqueeze(0)
    target = torch.tensor([target])
    target_var = target
    reference = process(input_var)
    [hevm.setInput(i, dat) for i, dat in enumerate([preprocess(input_var)])]
    timer = time.perf_counter_ns()
    hevm.run()
    timer = time.perf_counter_ns() -timer
    res = hevm.getOutput()
    res = postprocess(res, reference)
    err = res - reference 
    # print(res)
    # print(reference)
    rms = np.sqrt( np.sum(err*err) / res.shape[-1])
    hevm.printer(timer/pow(10, 9), rms)
