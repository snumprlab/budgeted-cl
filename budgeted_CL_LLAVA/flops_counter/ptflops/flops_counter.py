'''
Copyright (C) 2019-2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys

import torch.nn as nn

from .pytorch_engine import get_flops_pytorch
from .pytorch_engine_llava import get_flops_pytorch_llava
from .utils import flops_to_string, params_to_string


def get_model_complexity_info(model_or_trainer, input_res,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={}, backend='pytorch',
                              flops_units=None, param_units=None,
                              output_precision=2, criterion=None, original_opt=None, opt_name=None, lr=None, llava=False):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    # assert isinstance(model, nn.Module)

    if backend == 'pytorch':
        
        if llava:
            flops_dict = get_flops_pytorch_llava(model_or_trainer, input_res,
                                            print_per_layer_stat,
                                            input_constructor, ost,
                                            verbose, ignore_modules,
                                            custom_modules_hooks,
                                            output_precision=output_precision,
                                            flops_units=flops_units,
                                            param_units=param_units)
        else:
            flops_dict = get_flops_pytorch(model_or_trainer, input_res,
                                            print_per_layer_stat,
                                            input_constructor, ost,
                                            verbose, ignore_modules,
                                            custom_modules_hooks,
                                            output_precision=output_precision,
                                            flops_units=flops_units,
                                            param_units=param_units)
        # print(flops_dict)
    else:
        raise ValueError('Wrong backend name')

    return flops_dict

    # if as_strings:
    #     forward_flops_string, backward_flops_string, params_string, fc_params_string, buffers_string = make_string(forward_flops_count, backward_flops_count, params_count, fc_params_count, buffers_count)
    #     initial_forward_flops_string, initial_backward_flops_string, initial_params_string, _, _ = make_string(initial_forward_flops_count, initial_backward_flops_count, initial_params_count)
    #     group1_block0_forward_flops_string, group1_block0_backward_flops_string, group1_block0_params_string, _, _ = make_string(group1_block0_forward_flops_count, group1_block0_backward_flops_count, group1_block0_params_count)
    #     group1_block1_forward_flops_string, group1_block1_backward_flops_string, group1_block1_params_string, _, _ = make_string(group1_block1_forward_flops_count, group1_block1_backward_flops_count, group1_block1_params_count)
    #     group2_block0_forward_flops_string, group2_block0_backward_flops_string, group2_block0_params_string, _, _ = make_string(group2_block0_forward_flops_count, group2_block0_backward_flops_count, group2_block0_params_count)
    #     group2_block1_forward_flops_string, group2_block1_backward_flops_string, group2_block1_params_string, _, _ = make_string(group2_block1_forward_flops_count, group2_block1_backward_flops_count, group2_block1_params_count)
    #     group3_block0_forward_flops_string, group3_block0_backward_flops_string, group3_block0_params_string, _, _ = make_string(group3_block0_forward_flops_count, group3_block0_backward_flops_count, group3_block0_params_count)
    #     group3_block1_forward_flops_string, group3_block1_backward_flops_string, group3_block1_params_string, _, _ = make_string(group3_block1_forward_flops_count, group3_block1_backward_flops_count, group3_block1_params_count)
    #     group4_block0_forward_flops_string, group4_block0_backward_flops_string, group4_block0_params_string, _, _ = make_string(group4_block0_forward_flops_count, group4_block0_backward_flops_count, group4_block0_params_count)
    #     group4_block1_forward_flops_string, group4_block1_backward_flops_string, group4_block1_params_string, _, _ = make_string(group4_block1_forward_flops_count, group4_block1_backward_flops_count, group4_block1_params_count)
    #     fc_forward_flops_string, fc_backward_flops_string, fc_params_string, _, _ = make_string(fc_forward_flops_count, fc_backward_flops_count, fc_params_count)

    #     return [forward_flops_string, backward_flops_string, params_string, fc_params_string, buffers_string], \
    #         [initial_forward_flops_string, initial_backward_flops_string, initial_params_string], \
    #         [group1_block0_forward_flops_string, group1_block0_backward_flops_string, group1_block0_params_string], \
    #         [group1_block1_forward_flops_string, group1_block1_backward_flops_string, group1_block1_params_string], \
    #         [group2_block0_forward_flops_string, group2_block0_backward_flops_string, group2_block0_params_string], \
    #         [group2_block1_forward_flops_string, group2_block1_backward_flops_string, group2_block1_params_string], \
    #         [group3_block0_forward_flops_string, group3_block0_backward_flops_string, group3_block0_params_string], \
    #         [group3_block1_forward_flops_string, group3_block1_backward_flops_string, group3_block1_params_string], \
    #         [group4_block0_forward_flops_string, group4_block0_backward_flops_string, group4_block0_params_string], \
    #         [group4_block1_forward_flops_string, group4_block1_backward_flops_string, group4_block1_params_string], \
    #         [fc_forward_flops_string, fc_backward_flops_string, fc_params_string]
        

    # return [forward_flops_count, backward_flops_count, params_count, fc_params_count, buffers_count], \
    #     [initial_forward_flops_count, initial_backward_flops_count, initial_params_count], \
    #     [group1_block0_forward_flops_count, group1_block0_backward_flops_count, group1_block0_params_count], \
    #     [group1_block1_forward_flops_count, group1_block1_backward_flops_count, group1_block1_params_count], \
    #     [group2_block0_forward_flops_count, group2_block0_backward_flops_count, group2_block0_params_count], \
    #     [group2_block1_forward_flops_count, group2_block1_backward_flops_count, group2_block1_params_count], \
    #     [group3_block0_forward_flops_count, group3_block0_backward_flops_count, group3_block0_params_count], \
    #     [group3_block1_forward_flops_count, group3_block1_backward_flops_count, group3_block1_params_count], \
    #     [group4_block0_forward_flops_count, group4_block0_backward_flops_count, group4_block0_params_count], \
    #     [group4_block1_forward_flops_count, group4_block1_backward_flops_count, group4_block1_params_count], \
    #     [fc_forward_flops_count, fc_backward_flops_count, fc_params_count]
