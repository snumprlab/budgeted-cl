'''
Copyright (C) 2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''
from transformers.models.clip.modeling_clip import CLIPMLP
import numpy as np
import torch.nn as nn


def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    print("upsample input shape", input.shape, "output shape", output.shape)
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    # torch.numel() - Returns the total number of elements in the input tensor.
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)
    module.__back_flops__ += int(active_elements_count) # element들이 activation function 거치는 것이므로 그냥 active_element_count만큼만!


def linear_flops_counter_hook(module, input, output):
    input = input[0]

    if len(input.shape) > 2: ### matrix multiplication
        print("matrix input shape", input.shape, "output shape", output.shape)
    # (𝑛×𝑝) and (𝑝×𝑚) => 𝑛𝑚(2𝑝−1)
        n, p, m = input.shape[0], input.shape[1], input.shape[2]
        module.__flops__ += n * m * (2*p-1)
        module.__back_flops__ += (n * m * (2*p-1) * 2)
    else:
        # input : [1,512], output : [1,1] (앞의 1은 batch size, 뒤의 1은 fc layer의 output dimension)
        print("linear input shape", input.shape, "output shape", output.shape)
        # pytorch checks dimensions, so here we don't care much
        output_last_dim = output.shape[-1]
        bias_flops = output_last_dim if module.bias is not None else 0 # bias는 weight sum 다 끝내고 더해주는 애 이므로 output num만큼만 더해주기
        module.__flops__ += int(np.prod(input.shape) * output_last_dim + bias_flops)
        module.__back_flops__ += int(np.prod(input.shape) * output_last_dim + bias_flops) * 2 # back propagation, derivative term 2개이므로 *2


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    print("pool input shape", input.shape, "output shape", output.shape)
    
    module.__flops__ += int(np.prod(input.shape)) # pooling은 reduce dimension 위함 => 이 과정에서 모든 element들에 한번씩만 access함... 따라서 np.prod(input.shape)
    module.__back_flops__ += int(np.prod(input.shape)) # backward할때도 forward에서 방향만 반대이므로 forward와 flops 동일할 것


def bn_flops_counter_hook(module, input, output):
    input = input[0]
    print("bn input shape", input.shape, "output shape", output.shape)
    batch_flops = np.prod(input.shape) # 내가 *2 추가함
    '''
    if module.affine:
        batch_flops *= 2
    '''
    # 아래와 같이 선정한 이유는 batch_norm.py 참고
    module.__flops__ += int(batch_flops * 7)
    module.__back_flops__ += int(batch_flops * 15)


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]
    print("conv input shape", input.shape, "output shape", output.shape)
    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups
    # ex) kernel_dims = (7,7), output_dims = (112, 112)
    
    filters_per_channel = out_channels // groups

    conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel
    print("in_channels", in_channels, "out_channels", out_channels, "filters_per_channel", filters_per_channel, "kernel_dim", kernel_dims, "output_dims", output_dims)
    print()
    # conv fops = (K x K x input_channel) * (W X H X output_channel)
    # conv_per_position_flops = (K x K x input_channel)
    # active_elements_count = (W X H X filters_per_channel)
    # 이제 위의 2개를 곱한거에 batch_size만큼 곱해주면 conv flops
    # 거기에 conv 부분에 더해주는 bias가 있다면, out_channel(64) * out_dim(112*112) * batch_size 만큼의 bias를 더해주자
    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0

    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    conv_module.__flops__ += int(overall_flops)
    conv_module.__back_flops__ += (int(overall_flops) * 2) # derivative까지 해야하므로 *2


def rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0]*w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0]*w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size*3
        # last two hadamard product and add
        flops += rnn_module.hidden_size*3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size*4
        # two hadamard product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def rnn_flops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    If sigmoid and tanh are hard, only a comparison FLOPS should be accurate
    """
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def rnn_cell_flops_counter_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
# def multihead_attention_counter_hook(module, args, kwargs, output):
    # print(kwargs)
    # input = kwargs["images"]
    input = input[0]
    flops = 0
    print("!!mlp input shape", input.shape, "output shape", output.shape)
    q, k, v = input

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    flops = 0

    # Q scaling
    flops += qlen * qdim

    # Initial projections
    flops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

    if multihead_attention_module.in_proj_bias is not None:
        flops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_flops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    flops += qlen * vdim * (vdim + 1)

    flops *= batch_size
    multihead_attention_module.__flops__ += int(flops)


CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flops_counter_hook,
    nn.Conv2d: conv_flops_counter_hook,
    nn.Conv3d: conv_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_flops_counter_hook,
    nn.AvgPool1d: pool_flops_counter_hook,
    nn.AvgPool2d: pool_flops_counter_hook,
    nn.MaxPool2d: pool_flops_counter_hook,
    nn.MaxPool3d: pool_flops_counter_hook,
    nn.AvgPool3d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_flops_counter_hook,
    nn.BatchNorm2d: bn_flops_counter_hook,
    nn.BatchNorm3d: bn_flops_counter_hook,

    nn.InstanceNorm1d: bn_flops_counter_hook,
    nn.InstanceNorm2d: bn_flops_counter_hook,
    nn.InstanceNorm3d: bn_flops_counter_hook,
    nn.GroupNorm: bn_flops_counter_hook,
    # FC
    nn.Linear: linear_flops_counter_hook,
    # Upscale
    nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_flops_counter_hook,
    nn.ConvTranspose2d: conv_flops_counter_hook,
    nn.ConvTranspose3d: conv_flops_counter_hook,
    # RNN
    nn.RNN: rnn_flops_counter_hook,
    nn.GRU: rnn_flops_counter_hook,
    nn.LSTM: rnn_flops_counter_hook,
    nn.RNNCell: rnn_cell_flops_counter_hook,
    nn.LSTMCell: rnn_cell_flops_counter_hook,
    nn.GRUCell: rnn_cell_flops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook,
    # CLIPMLP: multihead_attention_counter_hook
}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_flops_counter_hook
