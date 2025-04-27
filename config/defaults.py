from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# BertForMaskedLM
# -----------------------------------------------------------------------------
_C.Fusion = CN()
_C.Fusion.architectures = ["BertForMaskedLM" ]
_C.Fusion.attention_probs_dropout_prob = 0.1
_C.Fusion.hidden_act = "gelu"
_C.Fusion.hidden_dropout_prob = 0.1
_C.Fusion.hidden_size = 768
_C.Fusion.initializer_range = 0.02
_C.Fusion.intermediate_size = 3072
_C.Fusion.layer_norm_eps = 1e-12
_C.Fusion.max_position_embeddings = 512
_C.Fusion.model_type = "bert"
_C.Fusion.num_attention_heads = 12
_C.Fusion.num_hidden_layers = 12
_C.Fusion.pad_token_id = 0
_C.Fusion.type_vocab_size = 2
_C.Fusion.vocab_size = 30522
_C.Fusion.encoder_width = 768
_C.Fusion.add_cross_attention = False
_C.Fusion.use_cache = False
_C.Fusion.gradient_checkpointing = False
_C.Fusion.fusion_layers = 6
_C.Fusion.text_decode_layers = 12
_C.Fusion.stride_layer = 3


# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0,1'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False
_C.MODEL.CLASS_TOKEN = 4

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
_C.SOLVER = CN()
_C.SOLVER.SEED = 1234
_C.SOLVER.MARGIN = 0.3

_C.SOLVER.IMS_PER_BATCH = 64
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.BASE_LR = 0.000005
_C.SOLVER.LR_MIN = 1e-7
_C.SOLVER.WARMUP_METHOD = 'linear'
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.WEIGHT_DECAY_BIAS = 1e-4
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.MAX_EPOCHS = 300
_C.SOLVER.CHECKPOINT_PERIOD = 30
_C.SOLVER.LOG_PERIOD = 50
_C.SOLVER.EVAL_PERIOD = 30
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.WARMUP_LR_INIT = 0.00001
_C.SOLVER.WARMUP_ITERS = 20
_C.SOLVER.WARMUP_FACTOR = 0.1

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
