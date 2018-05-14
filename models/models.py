
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned' or (opt.dataset_mode == 'unaligned_tensor_with_label' and opt.cond_nc == 0))
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode in  ['aligned','aligned_depth'])
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'pix2pix_stl':
        assert(opt.dataset_mode == 'aligned_with_flow')
        from .pix2pix_stl_model import Pix2PixSTLModel
        model = Pix2PixSTLModel()
    elif opt.model == 'pix2pix_depth':
        assert(opt.dataset_mode == 'aligned_depth')
        from .pix2pix_stl_model import Pix2PixSTLModel
        model = Pix2PixSTLModel()
    elif opt.model == 'ftae':
        assert(opt.dataset_mode == 'aligned')
        from .ftae_model import FTAEModel
        model = FTAEModel()
    elif opt.model == 'ftae_flow':
        assert(opt.dataset_mode in  ['aligned','aligned_with_C'])
        from .ftae_flow_model import FTAEModel
        model = FTAEModel()
    elif opt.model == 'flow_refine':
        assert(opt.dataset_mode in  ['aligned','aligned_with_C'])
        from .flow_refine_model import FlowRefineModel
        model = FlowRefineModel()
    elif opt.model == 'image_refine':
        assert(opt.dataset_mode in  ['aligned','aligned_with_C'])
        from .image_refine_model import ImageRefineModel
        model = ImageRefineModel()
    elif opt.model == 'multi_view_flow':
        assert(opt.dataset_mode in  ['aligned','aligned_with_C','aligned_multi_view'])
        from .multi_view_model import MultiViewModel
        model = MultiViewModel()
    elif opt.model == 'multi_view_depth':
        assert(opt.dataset_mode in  ['aligned','aligned_with_C','aligned_multi_view'])
        from .multi_view_depth_model import MultiViewDepthModel
        model = MultiViewDepthModel()
    elif opt.model == 'multi_view_depth_random':
        assert(opt.dataset_mode in  ['aligned','aligned_with_C','aligned_multi_view_random'])
        from .multi_view_depth_random_model import MultiViewDepthModel
        model = MultiViewDepthModel()
    elif opt.model == 'multi_view_flow_random':
        assert(opt.dataset_mode in  ['aligned','aligned_with_C','aligned_multi_view_random'])
        from .multi_view_flow_random_model import MultiViewFlowModel
        model = MultiViewFlowModel()
    elif opt.model == 'appearance_flow':
        assert(opt.dataset_mode in  ['aligned','aligned_with_C','appearance_flow'])
        from .appearance_flow_model import AppearanceFlowModel
        model = AppearanceFlowModel()
    elif opt.model == 'unsupervised_depth':
        assert(opt.dataset_mode in  ['aligned','aligned_with_C'])
        from .unsupervised_depth_model import UnsupervisedDepthModel
        model = UnsupervisedDepthModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
