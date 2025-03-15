import logging

from methods.er_new import ER
from methods.baseline_new import BASELINE
from methods.sdp_new import SDP
from methods.der_new import DER
from methods.ewc_new import EWCpp
from methods.ours_new import Ours
from methods.mir_new import MIR
from methods.aser_new import ASER
from methods.bic_new import BiasCorrection
from methods.remind_new import REMIND
from methods.memo_new import MEMO
from methods.ocs import OCS
from methods.xder import XDER
from methods.er_LiDER import ER_LiDER
from methods.der_LiDER import DER_LiDER
from methods.xder_LiDER import XDER_LiDER
from methods.mgi_dvc import MGI_DVC
from methods.afec_new import AFEC
from methods.co2l import CO2L
from methods.zero_shot_clip import ZeroShotClip
from methods.cupl import CuPL
from methods.sus import SUS
from methods.VLM import VLM

logger = logging.getLogger()


def select_method(args, train_datalist, test_datalist, device, model_args = None, data_args = None, training_args = None, bnb_model_from_pretrained_args = None):
    kwargs = vars(args)
    if args.mode == "er":
        method = ER(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "VLM":
        method = VLM(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            model_args=model_args, 
            data_args=data_args, 
            args=args,
            bnb_model_from_pretrained_args=bnb_model_from_pretrained_args
        )
    elif args.mode == "bic":
        method = BiasCorrection(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "mir":
        method = MIR(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "baseline":
        method = BASELINE(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "aser":
        method = ASER(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "ewc":
        method = EWCpp(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    # elif args.mode == "gdumb":
    #     from methods.gdumb import GDumb
    #     method = GDumb(
    #         train_datalist=train_datalist,
    #         test_datalist=test_datalist,
    #         cls_dict=cls_dict,
    #         device=device,
    #         **kwargs,
    #     )
    # elif args.mode == "mir":
    #     method = MIR(
    #         train_datalist=train_datalist,
    #         test_datalist=test_datalist,
    #         cls_dict=cls_dict,
    #         device=device,
    #         **kwargs,
    #     )
    # elif args.mode == "clib":
    #     method = CLIB(
    #         train_datalist=train_datalist,
    #         test_datalist=test_datalist,
    #         cls_dict=cls_dict,
    #         device=device,
    #         **kwargs,
    #     )
    elif args.mode == "der":
        method = DER(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "sdp":
        method = SDP(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "remind":
        method = REMIND(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "memo":
        method = MEMO(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "ocs":
        method = OCS(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )

    elif args.mode == "xder":
        method = XDER(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "er_lider":
        method = ER_LiDER(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "der_lider":
        method = DER_LiDER(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == 'xder_lider':
        method = XDER_LiDER(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "afec":
        method = AFEC(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "mgi_dvc":
        method = MGI_DVC(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "co2l":
        method = CO2L(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "zs_clip":
        method = ZeroShotClip(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "cupl":
        method = CuPL(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "sus":
        method = SUS(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "ours":
        method = Ours(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    
    else:
        raise NotImplementedError("Choose the args.mode in [er, gdumb, rm, bic, ewc++, mir, clib]")

    return method
