import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_DIR = r'.\data\maps\train'
VAL_DIR = r'.\data\maps\val'
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"
OUTPUT_PATH = r'D:\ParasiticEggDatasets\sr_pix2pix'
REPLICATE_DIRECTORY = True
both_transform = A.Compose(
    [
        #A.RandomCrop(width=512, height=512),
        #A.RandomSizedBBoxSafeCrop(width=512, height=512, erosion_rate=0.2), 
        #A.Flip(p=0.5),
        #A.ShiftScaleRotate(shift_limit=0., scale_limit=0.05, rotate_limit=45, p=0.2),
        #A.RandomScale(),
        #A.Rotate()

     ], 
     additional_targets={"image0": "image"},
     #bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
)

transform_only_input = A.Compose(
    [
        A.MotionBlur(blur_limit = [7, 21], p=1.),
        A.ColorJitter(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_target = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

root_path = [r'C:\Users\pm15334\ra\ParasiticEggDetection\datasets\01-11-2021_canon']
root_path = [#"C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\01-11-2021_canon",
           #"C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\01-11-2021_ip13",
           "C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\01-11-2021_nikon_dsi2",
           "C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\12-7-2021_c",
           "C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\12-7-2021_s",
           "C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\25-9-2021_nikon",
           "C:\\Users\\pm15334\\ra\\ParasiticEggDetection\\datasets\\25-9-2021_sumsu"]
root_path = r'D:\ParasiticEggDatasets\datasets'
colour = True