# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union

from monai.networks.layers.factories import Norm, split_args
from monai.utils import has_option
from  ipdb import set_trace
import torch.nn as nn
from torch import unsqueeze
__all__ = ["get_norm_layer"]

# def get_norm_layer(name: Union[Tuple, str], spatial_dims: Optional[int] = 1, channels: Optional[int] = 1, num_domains: Optional[int] = 2):
#     """
#     Create a normalization layer instance.

#     For example, to create normalization layers:

#     .. code-block:: python

#         from monai.networks.layers import get_norm_layer

#         g_layer = get_norm_layer(name=("group", {"num_groups": 1}))
#         n_layer = get_norm_layer(name="instance", spatial_dims=2)

#     Args:
#         name: a normalization type string or a tuple of type string and parameters.
#         spatial_dims: number of spatial dimensions of the input.
#         channels: number of features/channels when the normalization layer requires this parameter
#             but it is not specified in the norm parameters.
#     """
#     norm_name, norm_args = split_args(name)
#     norm_type = Norm[norm_name, spatial_dims]
#     kw_args = dict(norm_args)
#     if has_option(norm_type, "num_features") and "num_features" not in kw_args:
#         kw_args["num_features"] = channels
#     if has_option(norm_type, "num_channels") and "num_channels" not in kw_args:
#         kw_args["num_channels"] = channels

#     return nn.ModuleList([norm_type(**kw_args) for _ in range(num_domains)])

class get_norm_layer(nn.Module):
    def __init__(self, 
        name: Union[Tuple, str], 
        spatial_dims: Optional[int] = 1, 
        channels: Optional[int] = 1, 
        num_domains: Optional[int] = 2):

        self.num_domains = num_domains
        super(get_norm_layer, self).__init__()
        norm_name, norm_args = split_args(name)
        norm_type = Norm[norm_name, spatial_dims]
        kw_args = dict(norm_args)
        if has_option(norm_type, "num_features") and "num_features" not in kw_args:
            kw_args["num_features"] = channels
        if has_option(norm_type, "num_channels") and "num_channels" not in kw_args:
            kw_args["num_channels"] = channels
        self.norm = nn.ModuleList([norm_type(**kw_args) for _ in range(self.num_domains)])
    def forward(self, x):
        bs = x.shape[0]
        mini_bs= int(bs/self.num_domains)
        a=0
        b=mini_bs
        for i, N in enumerate(self.norm):
            if i!=0:
                a=b
                b=(i+1)*b 
            x[a:b,:,:,:,:]=N(x[a:b,:,:,:,:])
        return x

if __name__ == "__main__":
    import torch
    x=torch.rand(4,1,5,5,5)
    #x=torch.cat((y,y),0)
    # print(x)
    N=get_norm_layer(name="batch",spatial_dims=3, channels=1, num_domains=2)
    N(x)
    # set_trace()
