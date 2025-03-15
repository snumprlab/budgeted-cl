import torch
import copy
import os
import numpy as np
from .retrain import SparseTraining
from .utils_pr import load_configs, canonical_name
from .utils_pr import weight_pruning, weight_growing

def prune_init(kwargs, model, logger=None, pre_defined_mask=None):
    global prune_algo, retrain

    if kwargs["sp_retrain"]:
        if kwargs["sp_prune_before_retrain"]:
            prune_harden(kwargs, model)

        prune_algo = None
        retrain = SparseTraining(kwargs, model, logger, pre_defined_mask)
        return


def prune_update(epoch=0, batch_idx=0):
    retrain.update_mask(epoch, batch_idx)


def prune_harden(kwargs, model, option=None):
    configs, prune_ratios = load_configs(model, kwargs["sp_config_file"], logger=None)

    for key in prune_ratios:
        print("prune_ratios[{}]:{}".format(key, prune_ratios[key]))

    # self.logger.info("Hardened weight sparsity: name, num_nonzeros, total_num, sparsity")
    print("Hardened weight sparsity: name, num_nonzeros, total_num, sparsity")
    first = True
    for (name, W) in model.named_parameters():
        if name not in prune_ratios:  # ignore layers that do not have rho
            continue
        cuda_pruned_weights = None
        prune_ratio = prune_ratios[name]
        if option == None:
            cuda_pruned_weights = prune_weight(kwargs, configs, name, W, prune_ratio, first)  # get sparse model in cuda
            first = False
        else:
            raise Exception("not implmented yet")
        W.data = cuda_pruned_weights.cuda().type(W.dtype)  # replace the data field in variable

        if kwargs["sp_admm_sparsity_type"] == "block":
            block = eval(kwargs["sp_admm_block"])
            if block[1] == -1:  # row pruning, need to delete corresponding bias
                bias_layer = name.replace(".weight", ".bias")
                with torch.no_grad():
                    bias = model.state_dict()[bias_layer]
                    bias_mask = torch.sum(W, 1)
                    bias_mask[bias_mask != 0] = 1
                    bias.mul_(bias_mask)
        elif kwargs["sp_admm_sparsity_type"] == "filter":
            if not "downsample" in name:
                bn_weight_name = name.replace("conv", "bn")
                bn_bias_name = bn_weight_name.replace("weight", "bias")
            else:
                bn_weight_name = name.replace("downsample.0", "downsample.1")
                bn_bias_name = bn_weight_name.replace("weight", "bias")

            print("removing bn {}, {}".format(bn_weight_name, bn_bias_name))
            # bias_layer_name = name.replace(".weight", ".bias")

            with torch.no_grad():
                bn_weight = model.state_dict()[bn_weight_name]
                bn_bias = model.state_dict()[bn_bias_name]
                # bias = self.model.state_dict()[bias_layer_name]

                mask = torch.sum(W, (1, 2, 3))
                mask[mask != 0] = 1
                bn_weight.mul_(mask)
                bn_bias.mul_(mask)
                # bias.data.mul_(mask)

        non_zeros = W.detach().cpu().numpy() != 0
        non_zeros = non_zeros.astype(np.float32)
        num_nonzeros = np.count_nonzero(non_zeros)
        total_num = non_zeros.size
        sparsity = 1 - (num_nonzeros * 1.0) / total_num
        print("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
        # self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))


def prune_weight(kwargs, configs, name, weight, prune_ratio, first):
    if prune_ratio == 0.0:
        return weight
    # if pruning too many items, just prune everything
    if prune_ratio >= 0.999:
        return weight * 0.0
    if kwargs["sp_admm_sparsity_type"] == "irregular_global":
        _, res = weight_pruning(kwargs, configs, name, weight, prune_ratio)
    else:
        sp_admm_sparsity_type_copy = copy.copy(kwargs["sp_admm_sparsity_type"])
        sparsity_type_list = (kwargs["sp_admm_sparsity_type"]).split("+")
        if len(sparsity_type_list) != 1: #multiple sparsity type
            print(sparsity_type_list)
            for i in range(len(sparsity_type_list)):
                sparsity_type = sparsity_type_list[i]
                print("* sparsity type {} is {}".format(i, sparsity_type))
                kwargs["sp_admm_sparsity_type"] = sparsity_type
                _, weight =  weight_pruning(kwargs, configs, name, weight, prune_ratio)
                kwargs["sp_admm_sparsity_type"] = sp_admm_sparsity_type_copy
                print(np.sum(weight.detach().cpu().numpy() != 0))
            return weight.to(weight.device).type(weight.dtype)
        else:
            _, res = weight_pruning(kwargs, configs, name, weight, prune_ratio)

    return res.to(weight.device).type(weight.dtype)

def prune_apply_masks():
    if retrain:
        retrain.apply_masks()
    else:
        return
        assert(False)

def prune_apply_masks_on_grads():
    if retrain:
        retrain.apply_masks_on_grads()
    else:
        return
        assert(False)

def show_mask_sparsity():
    if retrain:
        retrain.test_mask_sparsity()
    else:
        return
        assert(False)

def prune_apply_masks_on_grads_mix():
    if retrain:
        retrain.apply_masks_on_grads_mix()
    else:
        return
        assert(False)

def prune_apply_masks_on_grads_efficient():
    if retrain:
        retrain.apply_masks_on_grads_efficient()
    else:
        return
        assert(False)


def update_prune_ratio(kwargs, model, prune_ratios, global_sparsity):
    if kwargs["sp_predefine_global_weight_sparsity_dir"] is not None:
        # use layer sparsity in a predefined sparse model to override prune_ratios
        print("=> loading checkpoint for keep ratio: {}".format(kwargs["sp_predefine_global_weight_sparsity_dir"]))

        assert os.path.exists(kwargs["sp_predefine_global_weight_sparsity_dir"]), "\n\n * Error, pre_defined sparse mask model not exist!"

        checkpoint = torch.load(kwargs["sp_predefine_global_weight_sparsity_dir"], map_location="cuda")
        model_state = checkpoint["state_dict"]
        for name, weight in model_state.items():
            if (canonical_name(name) not in prune_ratios.keys()) and (name not in prune_ratios.keys()):
                continue
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            non_zero = np.sum(weight.cpu().detach().numpy() != 0)
            new_prune_ratio = float(zeros / (zeros + non_zero))
            prune_ratios[name] = new_prune_ratio
        return prune_ratios

    total_size = 0
    for name, W in (model.named_parameters()):
        if (canonical_name(name) not in prune_ratios.keys()) \
                and (name not in prune_ratios.keys()):
            continue
        total_size += W.data.numel()
    to_prune = np.zeros(total_size)
    index = 0
    for name, W in (model.named_parameters()):
        if (canonical_name(name) not in prune_ratios.keys()) \
                and (name not in prune_ratios.keys()):
            continue
        size = W.data.numel()
        to_prune[index:(index+size)] = W.data.clone().cpu().view(-1).abs().numpy()
        index += size
    #sorted_to_prune = np.sort(to_prune)
    threshold = np.percentile(to_prune, global_sparsity*100)

    # update prune_ratios key-value pairs
    total_zeros = 0
    for name, W in (model.named_parameters()):
        if (canonical_name(name) not in prune_ratios.keys()) \
                and (name not in prune_ratios.keys()):
            continue
        size = W.data.numel()
        np_W_abs = W.detach().cpu().abs().numpy()
        new_prune_ratio = float(np.sum(np_W_abs < threshold))/size
        if new_prune_ratio >= 0.999:
            new_prune_ratio = 0.99

        total_zeros += float(np.sum(np_W_abs < threshold))

        prune_ratios[name] = new_prune_ratio

    print("Updated prune_ratios:")
    for key in prune_ratios:
        print("prune_ratios[{}]:{}".format(key,prune_ratios[key]))
    total_sparsity = total_zeros / total_size
    print("Total sparsity:{}".format(total_sparsity))

    return prune_ratios


def prune_print_sparsity(model=None, logger=None, show_sparse_only=False, compressed_view=False):
    if model is None:
        if prune_algo:
            model = prune_algo.model
        elif retrain:
            model = retrain.model
        else:
            return
    if logger:
        p = logger.info
    elif prune_algo:
        p = prune_algo.logger.info
    elif retrain:
        p = retrain.logger.info
    else:
        p = print

    if show_sparse_only:
        print("The sparsity of all params (>0.01): num_nonzeros, total_num, sparsity")
        for (name, W) in model.named_parameters():
            non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
            num_nonzeros = np.count_nonzero(non_zeros)
            total_num = non_zeros.size
            sparsity = 1 - (num_nonzeros * 1.0) / total_num
            if sparsity > 0.01:
                print("{}, {}, {}, {}, {}".format(name, non_zeros.shape, num_nonzeros, total_num, sparsity))
        return

    if compressed_view is True:
        total_w_num = 0
        total_w_num_nz = 0
        for (name, W) in model.named_parameters():
            if "weight" in name:
                non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
                num_nonzeros = np.count_nonzero(non_zeros)
                total_w_num_nz += num_nonzeros
                total_num = non_zeros.size
                total_w_num += total_num

        sparsity = 1 - (total_w_num_nz * 1.0) / total_w_num
        print("The sparsity of all params with 'weights' in its name: num_nonzeros, total_num, sparsity")
        print("{}, {}, {}".format(total_w_num_nz, total_w_num, sparsity))
        return

    print("The sparsity of all parameters: name, num_nonzeros, total_num, shape, sparsity")
    for (name, W) in model.named_parameters():
        non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
        num_nonzeros = np.count_nonzero(non_zeros)
        total_num = non_zeros.size
        sparsity = 1 - (num_nonzeros * 1.0) / total_num
        print("{}: {}, {}, {}, [{}]".format(name, str(num_nonzeros), str(total_num), non_zeros.shape, str(sparsity)))





