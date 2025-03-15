'''
Copyright (C) 2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys
from functools import partial
import copy
import torch
import torch.nn as nn

from .pytorch_ops import CUSTOM_MODULES_MAPPING, MODULES_MAPPING
from .utils import flops_to_string, params_to_string
from utils.train_utils import select_optimizer

def hook_fn(self, i, o):
    print("len i", len(i))
    print(i[0])
    print(i[1])
    print(i[2])
    print("i[0]", i[0].shape)
    print("i[1]", i[1].shape)
    print("i[2]", i[2].shape)
    print("output")
    print(o)

def get_flops_pytorch(model, input_res,
                      print_per_layer_stat=False,
                      input_constructor=None, ost=sys.stdout,
                      verbose=False, ignore_modules=[],
                      custom_modules_hooks={},
                      output_precision=3,
                      flops_units='GMac',
                      param_units='M'):
    
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks

    flops_model = add_flops_counting_methods(model) # counting 할 수 있도록 model 설정해주는 것
    flops_model.train()
    flops_model.start_flops_count(ost=ost, verbose=verbose, ignore_list=ignore_modules)

    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty((1, *input_res),
                                             dtype=next(flops_model.parameters()).dtype,
                                             device=next(flops_model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

        _ = flops_model(batch)
    
    forward_flops_count, backward_flops_count, params_count, fc_params_count, buffers_count = flops_model.compute_average_flops_cost()
    print("!!!!!!!!!!!")
    print("forward_flops_count", forward_flops_count, "backward_flops_count", backward_flops_count)
    print("!!!!!!!!!!!")
    
    if print_per_layer_stat:
        print_model_with_flops(
            flops_model,
            forward_flops_count,
            backward_flops_count,
            params_count,
            buffers_count,
            fc_params_count,
            ost=ost,
            flops_units=flops_units,
            param_units=param_units,
            precision=output_precision
        )
    flops_dict = return_layerwise_flops(
            flops_model,
            forward_flops_count,
            backward_flops_count,
            params_count,
            buffers_count,
            fc_params_count,
            ost=ost,
            flops_units=flops_units,
            param_units=param_units,
            precision=output_precision
        )
    flops_model.stop_flops_count()
    CUSTOM_MODULES_MAPPING = {}

    return flops_dict
    # return [forward_flops_count, backward_flops_count, params_count, fc_params_count, buffers_count]


def accumulate_back_flops(self):
    if is_supported_instance(self):
        return self.__back_flops__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_back_flops()
        return sum

def accumulate_flops(self):
    if is_supported_instance(self):
        return self.__flops__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_flops()
        return sum


def return_layerwise_flops(model, total_foward_flops, total_backward_flops, total_params, fc_params, total_buffers, flops_units='GMac',
                           param_units='M', precision=3, ost=sys.stdout):
    if total_foward_flops < 1:
        total_foward_flops = 1
    if total_backward_flops < 1:
        total_backward_flops = 1
    if total_params < 1:
        total_params = 1
    if fc_params < 1:
        fc_params = 1
    if total_buffers < 1:
        total_buffers = 1

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__, self.__fc_params__, self.__buffers__
        else:
            sum = 0
            fc_sum = 0
            buffer_sum = 0
            for m in self.children():
                param, fc_param, buffer = m.accumulate_params()
                sum += param
                fc_sum += fc_param
                buffer_sum += buffer
            return sum, fc_sum, buffer_sum

    def flops_repr(self):
        accumulated_params_num, accumulated_fc_params_num, accumulated_buffers_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops() / model.__batch_counter__
        accumulated_back_flops_cost = self.accumulate_back_flops() / model.__batch_counter__
        return ', '.join([params_to_string(accumulated_params_num,
                                           units=param_units, precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          params_to_string(accumulated_fc_params_num,
                                           units=param_units, precision=precision),
                          '{:.3%} fc_Params'.format(accumulated_fc_params_num / total_params),
                          params_to_string(accumulated_buffers_num,
                                           units=param_units, precision=precision),
                          '{:.3%} buffers'.format(accumulated_buffers_num / total_params),
                          flops_to_string(accumulated_flops_cost,
                                          units=flops_units, precision=precision),
                          '{:.3%} Forward MACs'.format(accumulated_flops_cost / total_foward_flops),
                          self.original_extra_repr(),
                          flops_to_string(accumulated_back_flops_cost,
                                          units=flops_units, precision=precision),
                          '{:.3%} Backward MACs'.format(accumulated_back_flops_cost / total_backward_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_back_flops = accumulate_back_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops
        if hasattr(m, 'accumulate_back_flops'):
            del m.accumulate_back_flops

    flops_dict = dict()
    
    model.apply(add_extra_repr)
    for name, module in model.named_modules():
        flops_dict[name] = {'forward_flops':module.accumulate_flops() / model.__batch_counter__, 'backward_flops':module.accumulate_back_flops() / model.__batch_counter__}
    model.apply(del_extra_repr)
    return flops_dict


def print_model_with_flops(model, total_foward_flops, total_backward_flops, total_params, fc_params, total_buffers, flops_units='GMac',
                           param_units='M', precision=3, ost=sys.stdout):
    if total_foward_flops < 1:
        total_foward_flops = 1
    if total_backward_flops < 1:
        total_backward_flops = 1
    if total_params < 1:
        total_params = 1
    if fc_params < 1:
        fc_params = 1
    if total_buffers < 1:
        total_buffers = 1

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__, self.__fc_params__, self.__buffers__
        else:
            sum = 0
            fc_sum = 0
            buffer_sum = 0
            for m in self.children():
                param, fc_param, buffer = m.accumulate_params()
                sum += param
                fc_sum += fc_param
                buffer_sum += buffer
            return sum, fc_sum, buffer_sum

    def flops_repr(self):
        accumulated_params_num, accumulated_fc_params_num, accumulated_buffers_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops() / model.__batch_counter__
        accumulated_back_flops_cost = self.accumulate_back_flops() / model.__batch_counter__
        return ', '.join([params_to_string(accumulated_params_num,
                                           units=param_units, precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          params_to_string(accumulated_fc_params_num,
                                           units=param_units, precision=precision),
                          '{:.3%} fc_Params'.format(accumulated_fc_params_num / total_params),
                          params_to_string(accumulated_buffers_num,
                                           units=param_units, precision=precision),
                          '{:.3%} buffers'.format(accumulated_buffers_num / total_params),
                          flops_to_string(accumulated_flops_cost,
                                          units=flops_units, precision=precision),
                          '{:.3%} Forward MACs'.format(accumulated_flops_cost / total_foward_flops),
                          self.original_extra_repr(),
                          flops_to_string(accumulated_back_flops_cost,
                                          units=flops_units, precision=precision),
                          '{:.3%} Backward MACs'.format(accumulated_back_flops_cost / total_backward_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_back_flops = accumulate_back_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops
        if hasattr(m, 'accumulate_back_flops'):
            del m.accumulate_back_flops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)
    model.apply(del_extra_repr)

def print_model_with_flops_sub(model, total_foward_flops, total_backward_flops, total_params, flops_units='GMac', param_units='M', precision=3, ost=sys.stdout):
    if total_foward_flops < 1:
        total_foward_flops = 1
    if total_backward_flops < 1:
        total_backward_flops = 1
    if total_params < 1:
        total_params = 1

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                param = m.accumulate_params()
                sum += param
            return sum

    def flops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops() / model.__batch_counter__
        accumulated_back_flops_cost = self.accumulate_back_flops() / model.__batch_counter__
        return ', '.join([params_to_string(accumulated_params_num,
                                           units=param_units, precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          flops_to_string(accumulated_flops_cost,
                                          units=flops_units, precision=precision),
                          '{:.3%} Forward MACs'.format(accumulated_flops_cost / total_foward_flops),
                          self.original_extra_repr(),
                          flops_to_string(accumulated_back_flops_cost,
                                          units=flops_units, precision=precision),
                          '{:.3%} Backward MACs'.format(accumulated_back_flops_cost / total_backward_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_back_flops = accumulate_back_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops
        if hasattr(m, 'accumulate_back_flops'):
            del m.accumulate_back_flops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)
    model.apply(del_extra_repr)



def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fc_params_num = sum(p.numel() for name, p in model.named_parameters() if 'fc' in name)
    buffers_num = sum(p.numel() for name, p in model.named_buffers())
    
    return params_num, fc_params_num, buffers_num

def add_backward_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_backward_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
                                                    net_main_module)
    net_main_module.reset_flops_count()

    return net_main_module

def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
                                                    net_main_module)
    net_main_module.reset_flops_count()

    # for layer별 flops count
    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    for m in self.modules():
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_back_flops = accumulate_back_flops.__get__(m)

    forward_flops_sum = self.accumulate_flops()
    backward_flops_sum = self.accumulate_back_flops()

    for m in self.modules():
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops
        if hasattr(m, 'accumulate_back_flops'):
            del m.accumulate_back_flops

    params_sum, fc_params_sum, buffers_sum = get_model_parameters_number(self)
    '''
    print("__batch_counter__", self.__batch_counter__)
    print("forward_flops_sum", forward_flops_sum)
    print("backward_flops_sum", backward_flops_sum)
    '''

    # layer별로 flops 구현할 때 batch_counter = 1로 만들어주어야 error 발생 X
    # print("self.__batch_counter__")
    # print(self.__batch_counter__)
    self.__batch_counter__=1
    return forward_flops_sum / self.__batch_counter__, backward_flops_sum / self.__batch_counter__, params_sum, fc_params_sum, buffers_sum

def start_backward_flops_count(self, **kwargs):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_backward_batch_counter_hook_function(self)
    seen_types = set()

    def add_flops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_backward_hook(
                                        CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_backward_hook(MODULES_MAPPING[type(module)])
            module.__flops_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
               not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_flops_counter_hook_function, **kwargs))

def start_flops_count(self, **kwargs):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_flops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                                        CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__flops_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
               not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    # add_flops_counter_hook_function에 kwargs를 넣어줌으로써, 각각의 용도에 맞게 사용할 수 있도록!
    
    '''
    def power(base, exponent):
        return base ** exponent

    square = partial(power, exponent=2)
    cube = partial(power, exponent=3)

    위와 같이 만들어줄 수 있다!
    '''
    
    # self.apply는 아래와 같이 weight initialize 할 때 자주 쓰인다.
    '''
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    
    net.apply(init_weights)
    '''
    self.apply(partial(add_flops_counter_hook_function, **kwargs))


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)
    self.apply(remove_flops_counter_variables)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions
def batch_counter_hook(module, input, output):
    print("input shape", len(input[0]))
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0

def add_backward_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return
    handle = module.register_backward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle

def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__') or hasattr(module, '__params__') or hasattr(module, '__back_flops__'):
            print('Warning: variables __flops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' ptflops can affect your code!')
            module.__ptflops_backup_flops__ = module.__flops__
            module.__ptflops_backup_back_flops__ = module.__back_flops__
            module.__ptflops_backup_params__ = module.__params__
        module.__flops__ = 0
        module.__back_flops__ = 0
        module.__params__, module.__fc_params__, module.__buffers__ = get_model_parameters_number(module)


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


def remove_flops_counter_variables(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__'):
            del module.__flops__
            if hasattr(module, '__ptflops_backup_flops__'):
                module.__flops__ = module.__ptflops_backup_flops__
        if hasattr(module, '__params__'):
            del module.__params__
            if hasattr(module, '__ptflops_backup_params__'):
                module.__params__ = module.__ptflops_backup_params__
        if hasattr(module, '__back_flops__'):
            del module.__back_flops__
            if hasattr(module, '__ptflops_backup_back_flops__'):
                module.__back_flops__ = module.__ptflops_backup_back_flops__
