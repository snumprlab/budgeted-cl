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
from methods.SparCL import SparCL
from methods.trire import TriRE
from methods.er_ccldc_new import ER_CCLDC
from methods.der_ccldc_new import DER_CCLDC
from methods.cama_new import CAMA
from methods.ewc_new_ccldc import EWCpp_CCLDC
from methods.egeria import EGERIA
logger = logging.getLogger()


def select_method(args, train_datalist, test_datalist, device):
    kwargs = vars(args)
    if args.mode == "er":
        method = ER(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
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
    elif args.mode == "egeria":
        method = EGERIA(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "er_ccldc":
        method = ER_CCLDC(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "der_ccldc":
        method = DER_CCLDC(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "ewc_ccldc":
        method = EWCpp_CCLDC(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
        
    elif args.mode == "cama":
        method = CAMA(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "trire":
        method = TriRE(
            train_datalist=train_datalist,
            test_datalist=test_datalist,
            device=device,
            **kwargs,
        )
    elif args.mode == "sparcl":
        method = SparCL(
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
