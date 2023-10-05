# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Model and data parallel groups."""

import torch

from .utils import ensure_divisibility


# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None

'''
最后，你可能想问，为什么还有一个embedding_group?
在GPT类模型中，输入层和输出层共享一个word_embedding（可参见第一部分中codegeex的架构图）。
因此，在计算完梯度，更新embedding权重前，输入和输出层需要进行通讯，保证word_embedding完全一致。
也即PP组中的第一个和最后一个进程需要通讯。我们知道设置进程子组的目的就是进一步划分通讯组，因此这里再添加一个embedding_group。
'''
# Embedding group.
_EMBEDDING_GROUP = None

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage
_PIPELINE_GLOBAL_RANKS = None


def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def initialize_model_parallel(
    tensor_model_parallel_size_=1,
    pipeline_model_parallel_size_=1,
    virtual_pipeline_model_parallel_size_=None,
):
    """
    Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used to parallelize model tensor.
        pipeline_model_parallel_size: number of GPUs used to parallelize model pipeline.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
        2 model-parallel group:
        [g0, g1, g4, g5, g8, g9, g12, g13], [g2, g3, g6, g7, g10, g8, g14, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    if torch.distributed.get_rank() == 0:
        print(
            "> initializing tensor model parallel with size {}".format(
                tensor_model_parallel_size_
            )
        )
        print( # 打印出流水线模型并行的度
            "> initializing pipeline model parallel with size {}".format(
                pipeline_model_parallel_size_
            )
        )
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized() # 确保torch已经做了分布式初始化
    world_size = torch.distributed.get_world_size() # 得到全局进程的总数
    tensor_model_parallel_size = min(tensor_model_parallel_size_, world_size)
    pipeline_model_parallel_size = min(pipeline_model_parallel_size_, world_size)

    ensure_divisibility( # 后者表示一个完整模型所占的gpu数，我们要保证前者能被后者整除
        world_size, tensor_model_parallel_size * pipeline_model_parallel_size
    )
    # 在codegeex中，TP_size=8, PP_size=1，world_size = 1536，因此DP_size是1536/(8*1) = 192
    data_parallel_size = world_size // ( # 根据TP_size和PP_size，求出DP_size
        tensor_model_parallel_size * pipeline_model_parallel_size
    )

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size # TP的组数
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size # PP的组数
    num_data_parallel_groups = world_size // data_parallel_size # DP的组数

    if virtual_pipeline_model_parallel_size_ is not None:
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = (
            virtual_pipeline_model_parallel_size_
        )

    rank = torch.distributed.get_rank() # 获取当前进程的全局rank

    # Build the data-parallel groups.（设置DP组）
    global _DATA_PARALLEL_GROUP # 保存DP组，如[[0,2], [1,3]...]，数字表示进进程的全局序号
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks) # 设置DP组
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group

    # Build the model-parallel groups.（设置MP组）
    global _MODEL_PARALLEL_GROUP # 保存MP组
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    for i in range(data_parallel_size):
        ranks = [
            data_parallel_group_ranks[i]
            for data_parallel_group_ranks in all_data_parallel_group_ranks
        ]
        group = torch.distributed.new_group(ranks) # 设置MP组
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.（设置TP组）
    global _TENSOR_MODEL_PARALLEL_GROUP # 保存TP组
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), "tensor model parallel group is already initialized"
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(
            i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size
        )
        group = torch.distributed.new_group(ranks) # 设置TP组
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).（设置PP组与embedding组）
    global _PIPELINE_MODEL_PARALLEL_GROUP # 设置PP组
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), "pipeline model parallel group is already initialized"
    global _EMBEDDING_GROUP
    assert _EMBEDDING_GROUP is None, "embedding group is already initialized"
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks) # 设置PP组
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
        else:
            embedding_ranks = ranks
        group = torch.distributed.new_group(embedding_ranks) # 设置embedding组
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
'''
总结来说，我们采用torch.distributed.new_group(ranks) 在进程大组下设置子组。
ranks是list of list，表示对进程序号的划分，例如设置DP组，则ranks为[[0,2], [1,3]...]，以此类推。
我们将划分结果存在全局变量中（例如_DATA_PARALLEL_GROUP），方便我们在后续切割模型时使用。
'''

def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP

# 同时，我们定义以下函数，使得对于任意一个进程，
# 我们都能查到它在DP/TP/PP组中的局部序号（local_rank），以及它对应的DP/TP/PP组的world_size。
# 这也是为后续切割模型使用:

# 这里展示和TP组相关的查询操作。其余组也是类推。详细代码一样都在megatron/mpu/initialize.py中
def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is not None
    ), "intra_layer_model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None
    ), "pipeline_model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, "embedding group is not initialized"
    return _EMBEDDING_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_model_parallel_world_size():
    assert (
        get_pipeline_model_parallel_world_size() == 1
    ), "legacy get_model_parallel_world_size is only supported if PP is disabled"
    return get_tensor_model_parallel_world_size()


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    # my_rank指的就是local_rank，例如[g2, g3]这一个TP组，rank为2，3；local_rank为0，1
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_model_parallel_rank():
    assert (
        get_pipeline_model_parallel_world_size() == 1
    ), "legacy get_model_parallel_rank is only supported if PP is disabled"
    return get_tensor_model_parallel_rank()


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = (
            get_virtual_pipeline_model_parallel_world_size()
        )
        if (
            virtual_pipeline_model_parallel_world_size is not None
            and get_virtual_pipeline_model_parallel_rank()
            != (virtual_pipeline_model_parallel_world_size - 1)
        ):
            return False
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1
    )


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_pipeline_model_parallel_first_rank():
    assert (
        _PIPELINE_GLOBAL_RANKS is not None
    ), "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    assert (
        _PIPELINE_GLOBAL_RANKS is not None
    ), "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    assert (
        _PIPELINE_GLOBAL_RANKS is not None
    ), "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    assert (
        _PIPELINE_GLOBAL_RANKS is not None
    ), "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def destroy_model_parallel():
    """Set the groups to none."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
