"""
This Python module provides an integrated pipeline for generating facial animations from a single
source image and an audio input. It leverages deep learning models and algorithms to process and 
animate the source image based on the provided audio cues. The pipeline includes preprocessing of 
the source image, extraction of 3D Morphable Model (3DMM) coefficients, audio processing to 
facial animation coefficients, and the rendering of the final animated video.

Key functionalities include:
- Cropping and extracting features from the source image to prepare it for animation.
- Converting audio input into facial animation coefficients using a deep neural network model.
- Animating the source image based on the audio-driven facial animation coefficients.
- Supporting additional reference inputs for eye blinking and head pose to enhance the realism of 
the animation.
- Batch processing capability for efficient handling of multiple inputs.
- Options for enhancing the quality of the generated face and optionally the background.
- Customizable animation settings such as pose style, batch size, image size, and expression scale.

Usage involves specifying paths to the source image, driven audio, and optional reference videos 
for eye blinking and pose, along with various parameters that control the animation process. The 
result is a rendered video that combines the source image and the driven audio into a coherent 
facial animation.

Example command line usage is provided at the end of the module for reference.

Dependencies:
- PyTorch for deep learning model operations.
- OpenCV and Pillow for image processing.
- NumPy for numerical operations.
- Other utility modules for file handling and argument parsing.
"""

import shutil
import os, sys
from time import  strftime
from argparse import ArgumentParser
from pathlib import Path

import torch

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

def folder_remove(folder):
	folder = str(folder)
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))

def main(args):
    #torch.backends.cudnn.enabled = False
    pic_path = args.source_image
    audio_path = args.driven_audio
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose
    result_dir = args.result_dir
    checkpoint_dir = args.checkpoint_dir
    size = args.size
    old_version = args.old_version
    preprocess = args.preprocess
    still = args.still
    face3dvis = args.face3dvis
    expression_scale = args.expression_scale
    preprocess = args.preprocess
    size = args.size
    enhancer = args.enhancer
    background_enhancer = args.background_enhancer


    save_dir = os.path.join(result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    aud_name = Path(audio_path).stem
    aud_folder = Path(save_dir)
    
    save_dir = aud_folder/'temp'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fn = aud_folder/f'{aud_name}.mp4'
    print(f'save filename {save_fn}')
    # os.makedirs(save_dir, exist_ok=True)

    current_root_path = os.path.split(sys.argv[0])[0]
    # current_root_path = "E:\\yt\\SadTalker"

    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size, old_version, preprocess)
    # print(f'sadtalker_paths: {sadtalker_paths}')
    
    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    
    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, preprocess, source_image_flag=True, pic_size=size)
    
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    if face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=expression_scale, still_mode=still, preprocess=preprocess, size=size)
    
    result, temp_dir = animate_from_coeff.generate(args, data, save_dir, pic_path, crop_info, enhancer=enhancer, background_enhancer=background_enhancer, preprocess=preprocess, img_size=size)
    shutil.move(result, save_fn)
    print('The generated video is named:', save_fn)
    folder_remove(temp_dir)

    # if not args.verbose:
    #     shutil.rmtree(save_dir)

    
if __name__ == '__main__':
    parser = ArgumentParser()  
    parser.add_argument("--driven_audio", default='./examples/driven_audio/imagine.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/00035-3227700327.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,  help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,  help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=512,  help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default="gfpgan", help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true") 
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks") 
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion") 
    parser.add_argument("--preprocess", default='extfull', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" ) 
    parser.add_argument("--verbose",action="store_true", help="saving the intermedia output or not" ) 
    parser.add_argument("--old_version",action="store_true", help="use the pth other than safetensor version" ) 


    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc',default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)

# Example:
# python inference.py --driven_audio "./examples/driven_audio/RD_Radio36_000.wav" --source_image "./examples/source_image/00035-3227700327.png" --pose_style 14 --enhancer gfpgan --size 512 --still
# 