# %%
COCO_CATS = ['__background__', 'airplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear',
             'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car',
             'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch', 'cow', 'cup', 'dining table', 'dog', 'donut',
             'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog',
             'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven',
             'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator', 'remote', 'sandwich', 'scissors',
             'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase',
             'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light',
             'train', 'truck', 'tv', 'umbrella', 'vase', 'wine glass', 'zebra']

# %%
from maskrcnn_benchmark.data.transforms.build import build_transforms
from yacs.config import CfgNode

# %%
voc_args = {'data_dir': 'data/VOCdevkit/VOC2007',
 'split': 'trainval', 
 'use_difficult': False, 
 'external_proposal': False, 
 'old_classes': [], 
 'new_classes': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow'], 
 'excluded_classes': ['diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'], 
 'is_train': True, 
 'cfg': CfgNode({'MODEL': CfgNode({'RPN_ONLY': False, 'MASK_ON': False, 'RETINANET_ON': False, 
                                    'KEYPOINT_ON': False, 'DEVICE': 'cuda', 
                                    'META_ARCHITECTURE': 'GeneralizedRCNN', 'CLS_AGNOSTIC_BBOX_REG': False,
                                    'WEIGHT': 'catalog://ImageNetPretrained/MSRA/R-50', 'SOURCE_WEIGHT': '', 
                                    'BACKBONE': CfgNode({'CONV_BODY': 'R-50-C4', 'FREEZE_CONV_BODY_AT': 2, 
                                                        'USE_GN': False, 'ALL_FREEZE': False, 'FPN_FREEZE': False
                                                        }),
                                    'FPN': CfgNode({'USE_GN': False, 'USE_RELU': False}), 
                                    'GROUP_NORM': CfgNode({'DIM_PER_GP': -1, 'NUM_GROUPS': 32, 'EPSILON': 1e-05}), 
                                    'RPN': CfgNode({'USE_FPN': False, 'ANCHOR_SIZES': (32, 64, 128, 256, 512), 
                                                    'ANCHOR_STRIDE': (16,), 'ASPECT_RATIOS': (0.5, 1.0, 2.0), 'STRADDLE_THRESH': 0, 
                                                    'FG_IOU_THRESHOLD': 0.7, 'BG_IOU_THRESHOLD': 0.3, 'BATCH_SIZE_PER_IMAGE': 256, 
                                                    'POSITIVE_FRACTION': 0.5, 'PRE_NMS_TOP_N_TRAIN': 12000, 'PRE_NMS_TOP_N_TEST': 6000, 
                                                    'POST_NMS_TOP_N_TRAIN': 2000, 'POST_NMS_TOP_N_TEST': 1000, 'NMS_THRESH': 0.7, 'MIN_SIZE': 0, 
                                                    'FPN_POST_NMS_TOP_N_TRAIN': 2000, 'FPN_POST_NMS_TOP_N_TEST': 2000, 'FPN_POST_NMS_PER_BATCH': True, 
                                                    'RPN_HEAD': 'SingleConvRPNHead', 'EXTERNAL_PROPOSAL': False, 'CONV_FREEZE': False, 'CLS_FREEZE': False, 'BBS_FREEZE': False
                                                    }), 
                                    'ROI_HEADS': CfgNode({'USE_FPN': False, 'FG_IOU_THRESHOLD': 0.5, 'BG_IOU_THRESHOLD': 0.5, 'BBOX_REG_WEIGHTS': (10.0, 10.0, 5.0, 5.0),
                                                        'BATCH_SIZE_PER_IMAGE': 512, 'POSITIVE_FRACTION': 0.25, 'FC_FREEZE': False, 'CLS_FREEZE': False, 'BBS_FREEZE': False, 
                                                        'CLS_OFFSET': False, 'BBS_OFFSET': False, 'SCORE_THRESH': 0.05, 'NMS': 0.5, 'DETECTIONS_PER_IMG': 100
                                                        }), 
                                    'ROI_BOX_HEAD': CfgNode({'FEATURE_EXTRACTOR': 'ResNet50Conv5ROIFeatureExtractor', 
                                                            'PREDICTOR': 'FastRCNNPredictor', 'POOLER_RESOLUTION': 7, 
                                                            'POOLER_SAMPLING_RATIO': 2, 'POOLER_SCALES': (0.0625,), 'NUM_CLASSES': 11,
                                                            'NAME_OLD_CLASSES': [], 
                                                            'NAME_NEW_CLASSES': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow'], 
                                                            'NAME_EXCLUDED_CLASSES': ['diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'], 
                                                            'MLP_HEAD_DIM': 1024, 'USE_GN': False, 'DILATION': 1, 'CONV_HEAD_DIM': 256, 'NUM_STACKED_CONVS': 4
                                                            }), 
                                    'ROI_MASK_HEAD': CfgNode({'FEATURE_EXTRACTOR': 'ResNet50Conv5ROIFeatureExtractor', 'PREDICTOR': 'MaskRCNNC4Predictor', 
                                                            'POOLER_RESOLUTION': 14, 'POOLER_SAMPLING_RATIO': 0, 'POOLER_SCALES': (0.0625,), 'MLP_HEAD_DIM': 1024, 
                                                            'CONV_LAYERS': (256, 256, 256, 256), 'RESOLUTION': 14, 'SHARE_BOX_FEATURE_EXTRACTOR': True, 
                                                            'POSTPROCESS_MASKS': False, 'POSTPROCESS_MASKS_THRESHOLD': 0.5, 'DILATION': 1, 'USE_GN': False
                                                            }),
                                    'ROI_KEYPOINT_HEAD': CfgNode({'FEATURE_EXTRACTOR': 'KeypointRCNNFeatureExtractor', 'PREDICTOR': 'KeypointRCNNPredictor', 'POOLER_RESOLUTION': 14, 
                                                                'POOLER_SAMPLING_RATIO': 0, 'POOLER_SCALES': (0.0625,), 'MLP_HEAD_DIM': 1024, 'CONV_LAYERS': (512, 512, 512, 512, 512, 512, 512, 512),
                                                                'RESOLUTION': 14, 'NUM_CLASSES': 17, 'SHARE_BOX_FEATURE_EXTRACTOR': True
                                                                }), 
                                    'RESNETS': CfgNode({'NUM_GROUPS': 1, 'WIDTH_PER_GROUP': 64, 'STRIDE_IN_1X1': True, 'TRANS_FUNC': 'BottleneckWithFixedBatchNorm', 
                                                        'STEM_FUNC': 'StemWithFixedBatchNorm', 'RES5_DILATION': 1, 'BACKBONE_OUT_CHANNELS': 1024, 
                                                        'RES2_OUT_CHANNELS': 256, 'STEM_OUT_CHANNELS': 64, 'STAGE_WITH_DCN': (False, False, False, False),
                                                        'WITH_MODULATED_DCN': False, 'DEFORMABLE_GROUPS': 1
                                                        }), 
                                    'RETINANET': CfgNode({'NUM_CLASSES': 81, 'ANCHOR_SIZES': (32, 64, 128, 256, 512), 
                                                        'ASPECT_RATIOS': (0.5, 1.0, 2.0), 'ANCHOR_STRIDES': (8, 16, 32, 64, 128), 
                                                        'STRADDLE_THRESH': 0, 'OCTAVE': 2.0, 'SCALES_PER_OCTAVE': 3, 'USE_C5': True, 
                                                        'NUM_CONVS': 4, 'BBOX_REG_WEIGHT': 4.0, 'BBOX_REG_BETA': 0.11, 'PRE_NMS_TOP_N': 1000,
                                                        'FG_IOU_THRESHOLD': 0.5, 'BG_IOU_THRESHOLD': 0.4, 'LOSS_ALPHA': 0.25, 'LOSS_GAMMA': 2.0,
                                                        'PRIOR_PROB': 0.01, 'INFERENCE_TH': 0.05, 'NMS_TH': 0.4
                                                        }), 
                                    'FBNET': CfgNode({'ARCH': 'default', 'ARCH_DEF': '', 'BN_TYPE': 'bn', 'SCALE_FACTOR': 1.0, 'WIDTH_DIVISOR': 1,
                                                    'DW_CONV_SKIP_BN': True, 'DW_CONV_SKIP_RELU': True, 'DET_HEAD_LAST_SCALE': 1.0, 'DET_HEAD_BLOCKS': [], 
                                                    'DET_HEAD_STRIDE': 0, 'KPTS_HEAD_LAST_SCALE': 0.0, 'KPTS_HEAD_BLOCKS': [], 'KPTS_HEAD_STRIDE': 0, 
                                                    'MASK_HEAD_LAST_SCALE': 0.0, 'MASK_HEAD_BLOCKS': [], 'MASK_HEAD_STRIDE': 0, 'RPN_HEAD_BLOCKS': 0, 'RPN_BN_TYPE': ''
                                                    })
                                    }), 
            'INPUT': CfgNode({'MIN_SIZE_TRAIN': (800,), 'MAX_SIZE_TRAIN': 1333, 'MIN_SIZE_TEST': 800, 'MAX_SIZE_TEST': 1333, 'FLIP_PROB_TRAIN': 0.5, 'PIXEL_MEAN': [102.9801, 115.9465, 122.7717], 
                                'PIXEL_STD': [1.0, 1.0, 1.0], 'TO_BGR255': True, 'BRIGHTNESS': 0.0, 'CONTRAST': 0.0, 'SATURATION': 0.0, 'HUE': 0.0}), 
            'DATASETS': CfgNode({'TRAIN': ('voc_rb_2007_trainval',), 'TEST': ('voc_2007_test',)}), 
            'DATALOADER': CfgNode({'NUM_WORKERS': 4, 'SIZE_DIVISIBILITY': 32, 'ASPECT_RATIO_GROUPING': True}), 
            'SOLVER': CfgNode({'MAX_ITER': 10000, 'BASE_LR': 0.005, 'BIAS_LR_FACTOR': 2, 'MOMENTUM': 0.9, 'WEIGHT_DECAY': 0.0001, 'WEIGHT_DECAY_BIAS': 0, 'GAMMA': 0.1, 'STEPS': (7500,), 
                                'WARMUP_FACTOR': 0.3333333333333333, 'WARMUP_ITERS': 500, 'WARMUP_METHOD': 'linear', 'CHECKPOINT_PERIOD': 2500, 'IMS_PER_BATCH': 4}), 
            'TEST': CfgNode({'EXPECTED_RESULTS': [], 'EXPECTED_RESULTS_SIGMA_TOL': 4, 'IMS_PER_BATCH': 4, 'DETECTIONS_PER_IMG': 100, 'COCO_ALPHABETICAL_ORDER': False}), 
            'OUTPUT_DIR': 'output/10-10/LR005_BS4_FILOD', 
            'TENSORBOARD_DIR': './tensorboardx', 
            'PATHS_CATALOG': '/mnt/disk1/tiennh/IOD/ABR_IOD/maskrcnn_benchmark/config/paths_catalog.py', 
            'INCREMENTAL': False, 
            'DIST': CfgNode({'ROI_ALIGN': False, 'TYPE': 'l2', 'RPN': False, 'FEAT': False, 'ATT': False, 'ALPHA': 0.0, 'BETA': 0.0, 'GAMMA': 1.0}), 
            'UCE_WEIGHT': 1.0, 
            'MEM_BUFF': None, 
            'MEM_TYPE': False, 
            'IS_SAMPLE': False, 
            'IS_FATHER': True, 
            'DTYPE': 'float32', 
            'CLS_PER_STEP': -1, 
            'AMP_VERBOSE': False}), 
#  'transforms':
}

# %%
cfg = voc_args['cfg']
transform = build_transforms(cfg, is_train=True)
voc_args['transform'] = transform
# %%
coco_args = {'root': 'coco/train2017', 'ann_file': 'coco/annotations/instances_train2017.json', 'transforms': }

# %%


# %%
{'data_dir': 'data/VOCdevkit/VOC2007',
 'split': 'trainval', 
 'use_difficult': False, 
 'external_proposal': False, 
 'old_classes': [], 
 'new_classes': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow'], 
 'excluded_classes': ['diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'], 
 'is_train': True, 
 'cfg': CfgNode({'MODEL': CfgNode({'RPN_ONLY': False, 'MASK_ON': False, 'RETINANET_ON': False, 
                                    'KEYPOINT_ON': False, 'DEVICE': 'cuda', 
                                    'META_ARCHITECTURE': 'GeneralizedRCNN', 'CLS_AGNOSTIC_BBOX_REG': False,
                                    'WEIGHT': 'catalog://ImageNetPretrained/MSRA/R-50', 'SOURCE_WEIGHT': '', 
                                    'BACKBONE': CfgNode({'CONV_BODY': 'R-50-C4', 'FREEZE_CONV_BODY_AT': 2, 
                                                        'USE_GN': False, 'ALL_FREEZE': False, 'FPN_FREEZE': False
                                                        }),
                                    'FPN': CfgNode({'USE_GN': False, 'USE_RELU': False}), 
                                    'GROUP_NORM': CfgNode({'DIM_PER_GP': -1, 'NUM_GROUPS': 32, 'EPSILON': 1e-05}), 
                                    'RPN': CfgNode({'USE_FPN': False, 'ANCHOR_SIZES': (32, 64, 128, 256, 512), 
                                                    'ANCHOR_STRIDE': (16,), 'ASPECT_RATIOS': (0.5, 1.0, 2.0), 'STRADDLE_THRESH': 0, 
                                                    'FG_IOU_THRESHOLD': 0.7, 'BG_IOU_THRESHOLD': 0.3, 'BATCH_SIZE_PER_IMAGE': 256, 
                                                    'POSITIVE_FRACTION': 0.5, 'PRE_NMS_TOP_N_TRAIN': 12000, 'PRE_NMS_TOP_N_TEST': 6000, 
                                                    'POST_NMS_TOP_N_TRAIN': 2000, 'POST_NMS_TOP_N_TEST': 1000, 'NMS_THRESH': 0.7, 'MIN_SIZE': 0, 
                                                    'FPN_POST_NMS_TOP_N_TRAIN': 2000, 'FPN_POST_NMS_TOP_N_TEST': 2000, 'FPN_POST_NMS_PER_BATCH': True, 
                                                    'RPN_HEAD': 'SingleConvRPNHead', 'EXTERNAL_PROPOSAL': False, 'CONV_FREEZE': False, 'CLS_FREEZE': False, 'BBS_FREEZE': False
                                                    }), 
                                    'ROI_HEADS': CfgNode({'USE_FPN': False, 'FG_IOU_THRESHOLD': 0.5, 'BG_IOU_THRESHOLD': 0.5, 'BBOX_REG_WEIGHTS': (10.0, 10.0, 5.0, 5.0),
                                                        'BATCH_SIZE_PER_IMAGE': 512, 'POSITIVE_FRACTION': 0.25, 'FC_FREEZE': False, 'CLS_FREEZE': False, 'BBS_FREEZE': False, 
                                                        'CLS_OFFSET': False, 'BBS_OFFSET': False, 'SCORE_THRESH': 0.05, 'NMS': 0.5, 'DETECTIONS_PER_IMG': 100
                                                        }), 
                                    'ROI_BOX_HEAD': CfgNode({'FEATURE_EXTRACTOR': 'ResNet50Conv5ROIFeatureExtractor', 
                                                            'PREDICTOR': 'FastRCNNPredictor', 'POOLER_RESOLUTION': 7, 
                                                            'POOLER_SAMPLING_RATIO': 2, 'POOLER_SCALES': (0.0625,), 'NUM_CLASSES': 11,
                                                            'NAME_OLD_CLASSES': [], 
                                                            'NAME_NEW_CLASSES': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow'], 
                                                            'NAME_EXCLUDED_CLASSES': ['diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'], 
                                                            'MLP_HEAD_DIM': 1024, 'USE_GN': False, 'DILATION': 1, 'CONV_HEAD_DIM': 256, 'NUM_STACKED_CONVS': 4
                                                            }), 
                                    'ROI_MASK_HEAD': CfgNode({'FEATURE_EXTRACTOR': 'ResNet50Conv5ROIFeatureExtractor', 'PREDICTOR': 'MaskRCNNC4Predictor', 
                                                            'POOLER_RESOLUTION': 14, 'POOLER_SAMPLING_RATIO': 0, 'POOLER_SCALES': (0.0625,), 'MLP_HEAD_DIM': 1024, 
                                                            'CONV_LAYERS': (256, 256, 256, 256), 'RESOLUTION': 14, 'SHARE_BOX_FEATURE_EXTRACTOR': True, 
                                                            'POSTPROCESS_MASKS': False, 'POSTPROCESS_MASKS_THRESHOLD': 0.5, 'DILATION': 1, 'USE_GN': False
                                                            }),
                                    'ROI_KEYPOINT_HEAD': CfgNode({'FEATURE_EXTRACTOR': 'KeypointRCNNFeatureExtractor', 'PREDICTOR': 'KeypointRCNNPredictor', 'POOLER_RESOLUTION': 14, 
                                                                'POOLER_SAMPLING_RATIO': 0, 'POOLER_SCALES': (0.0625,), 'MLP_HEAD_DIM': 1024, 'CONV_LAYERS': (512, 512, 512, 512, 512, 512, 512, 512),
                                                                'RESOLUTION': 14, 'NUM_CLASSES': 17, 'SHARE_BOX_FEATURE_EXTRACTOR': True
                                                                }), 
                                    'RESNETS': CfgNode({'NUM_GROUPS': 1, 'WIDTH_PER_GROUP': 64, 'STRIDE_IN_1X1': True, 'TRANS_FUNC': 'BottleneckWithFixedBatchNorm', 
                                                        'STEM_FUNC': 'StemWithFixedBatchNorm', 'RES5_DILATION': 1, 'BACKBONE_OUT_CHANNELS': 1024, 
                                                        'RES2_OUT_CHANNELS': 256, 'STEM_OUT_CHANNELS': 64, 'STAGE_WITH_DCN': (False, False, False, False),
                                                        'WITH_MODULATED_DCN': False, 'DEFORMABLE_GROUPS': 1
                                                        }), 
                                    'RETINANET': CfgNode({'NUM_CLASSES': 81, 'ANCHOR_SIZES': (32, 64, 128, 256, 512), 
                                                        'ASPECT_RATIOS': (0.5, 1.0, 2.0), 'ANCHOR_STRIDES': (8, 16, 32, 64, 128), 
                                                        'STRADDLE_THRESH': 0, 'OCTAVE': 2.0, 'SCALES_PER_OCTAVE': 3, 'USE_C5': True, 
                                                        'NUM_CONVS': 4, 'BBOX_REG_WEIGHT': 4.0, 'BBOX_REG_BETA': 0.11, 'PRE_NMS_TOP_N': 1000,
                                                        'FG_IOU_THRESHOLD': 0.5, 'BG_IOU_THRESHOLD': 0.4, 'LOSS_ALPHA': 0.25, 'LOSS_GAMMA': 2.0,
                                                        'PRIOR_PROB': 0.01, 'INFERENCE_TH': 0.05, 'NMS_TH': 0.4
                                                        }), 
                                    'FBNET': CfgNode({'ARCH': 'default', 'ARCH_DEF': '', 'BN_TYPE': 'bn', 'SCALE_FACTOR': 1.0, 'WIDTH_DIVISOR': 1,
                                                    'DW_CONV_SKIP_BN': True, 'DW_CONV_SKIP_RELU': True, 'DET_HEAD_LAST_SCALE': 1.0, 'DET_HEAD_BLOCKS': [], 
                                                    'DET_HEAD_STRIDE': 0, 'KPTS_HEAD_LAST_SCALE': 0.0, 'KPTS_HEAD_BLOCKS': [], 'KPTS_HEAD_STRIDE': 0, 
                                                    'MASK_HEAD_LAST_SCALE': 0.0, 'MASK_HEAD_BLOCKS': [], 'MASK_HEAD_STRIDE': 0, 'RPN_HEAD_BLOCKS': 0, 'RPN_BN_TYPE': ''
                                                    })
                                    }), 
            'INPUT': CfgNode({'MIN_SIZE_TRAIN': (800,), 'MAX_SIZE_TRAIN': 1333, 'MIN_SIZE_TEST': 800, 'MAX_SIZE_TEST': 1333, 'FLIP_PROB_TRAIN': 0.5, 'PIXEL_MEAN': [102.9801, 115.9465, 122.7717], 
                                'PIXEL_STD': [1.0, 1.0, 1.0], 'TO_BGR255': True, 'BRIGHTNESS': 0.0, 'CONTRAST': 0.0, 'SATURATION': 0.0, 'HUE': 0.0}), 
            'DATASETS': CfgNode({'TRAIN': ('voc_rb_2007_trainval',), 'TEST': ('voc_2007_test',)}), 
            'DATALOADER': CfgNode({'NUM_WORKERS': 4, 'SIZE_DIVISIBILITY': 32, 'ASPECT_RATIO_GROUPING': True}), 
            'SOLVER': CfgNode({'MAX_ITER': 10000, 'BASE_LR': 0.005, 'BIAS_LR_FACTOR': 2, 'MOMENTUM': 0.9, 'WEIGHT_DECAY': 0.0001, 'WEIGHT_DECAY_BIAS': 0, 'GAMMA': 0.1, 'STEPS': (7500,), 
                                'WARMUP_FACTOR': 0.3333333333333333, 'WARMUP_ITERS': 500, 'WARMUP_METHOD': 'linear', 'CHECKPOINT_PERIOD': 2500, 'IMS_PER_BATCH': 4}), 
            'TEST': CfgNode({'EXPECTED_RESULTS': [], 'EXPECTED_RESULTS_SIGMA_TOL': 4, 'IMS_PER_BATCH': 4, 'DETECTIONS_PER_IMG': 100, 'COCO_ALPHABETICAL_ORDER': False}), 
            'OUTPUT_DIR': 'output/10-10/LR005_BS4_FILOD', 
            'TENSORBOARD_DIR': './tensorboardx', 
            'PATHS_CATALOG': '/mnt/disk1/tiennh/IOD/ABR_IOD/maskrcnn_benchmark/config/paths_catalog.py', 
            'INCREMENTAL': False, 
            'DIST': CfgNode({'ROI_ALIGN': False, 'TYPE': 'l2', 'RPN': False, 'FEAT': False, 'ATT': False, 'ALPHA': 0.0, 'BETA': 0.0, 'GAMMA': 1.0}), 
            'UCE_WEIGHT': 1.0, 
            'MEM_BUFF': None, 
            'MEM_TYPE': False, 
            'IS_SAMPLE': False, 
            'IS_FATHER': True, 
            'DTYPE': 'float32', 
            'CLS_PER_STEP': -1, 
            'AMP_VERBOSE': False}), 
 'transforms': Compose(
    <maskrcnn_benchmark.data.transforms.transforms.ColorJitter object at 0x7fd79742dc50>
    <maskrcnn_benchmark.data.transforms.transforms.Resize object at 0x7fd79742d510>
    <maskrcnn_benchmark.data.transforms.transforms.RandomHorizontalFlip object at 0x7fd79742d590>
    <maskrcnn_benchmark.data.transforms.transforms.ToTensor object at 0x7fd79742d610>
    <maskrcnn_benchmark.data.transforms.transforms.Normalize object at 0x7fd797409f10>
)}

# %%
# coco
config_file = 'configs/coco/70-10/e2e_faster_rcnn_R_50_C4_4x.yaml'


# %%
# voc
config_file = 'configs/voc/10-10/e2e_faster_rcnn_R_50_C4_4x.yaml'




