from __future__ import print_function
import os
import torch
from torch.utils.ffi import create_extension


sources = ['src/pcl_losses.c']
headers = ['src/pcl_losses.h']
extra_objects = []
defines = []
with_cuda = False

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/pcl_losses_cuda.c']
    headers += ['src/pcl_losses_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
    extra_objects = ['src/pcl_losses_kernel.cu.o']
    extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.pcl_losses',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
