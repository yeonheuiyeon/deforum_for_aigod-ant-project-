from flask import Flask, request, json, jsonify,send_from_directory,abort
from werkzeug.utils import secure_filename
import whisper
import torch
import time
import re
import cv2
import numpy as np
import os
from os.path import isfile, join
import openai
import subprocess, time, gc, os, sys

app = Flask(__name__)

asrmodel = whisper.load_model('medium')
asrmodel.to(torch.device("cuda:3"))
transcribe_options = dict(task="transcribe")
translate_options = dict(task="translate")

def setup_environment():
    print_subprocess = False
    use_xformers_for_colab = True
    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'
    if 'google.colab' in str(ipy):
        print("..setting up environment")
        start_time = time.time()
        all_process = [
            ['pip', 'install', 'torch==1.12.1+cu113', 'torchvision==0.13.1+cu113', '--extra-index-url',
             'https://download.pytorch.org/whl/cu113'],
            ['pip', 'install', 'omegaconf==2.2.3', 'einops==0.4.1', 'pytorch-lightning==1.7.4', 'torchmetrics==0.9.3',
             'torchtext==0.13.1', 'transformers==4.21.2', 'kornia==0.6.7'],
            ['git', 'clone', 'https://github.com/deforum-art/deforum-stable-diffusion'],
            ['pip', 'install', 'accelerate', 'ftfy', 'jsonmerge', 'matplotlib', 'resize-right', 'timm', 'torchdiffeq',
             'scikit-learn'],
        ]
        for process in all_process:
            running = subprocess.run(process, stdout=subprocess.PIPE).stdout.decode('utf-8')
            if print_subprocess:
                print(running)
        with open('deforum-stable-diffusion/src/k_diffusion/__init__.py', 'w') as f:
            f.write('')
        sys.path.extend([
            'deforum-stable-diffusion/',
            'deforum-stable-diffusion/src',
        ])
        end_time = time.time()

        if use_xformers_for_colab:

            print("..installing xformers")

            all_process = [['pip', 'install', 'triton==2.0.0.dev20220701']]
            for process in all_process:
                running = subprocess.run(process, stdout=subprocess.PIPE).stdout.decode('utf-8')
                if print_subprocess:
                    print(running)

            v_card_name = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                         stdout=subprocess.PIPE).stdout.decode('utf-8')
            if 't4' in v_card_name.lower():
                name_to_download = 'T4'
            elif 'v100' in v_card_name.lower():
                name_to_download = 'V100'
            elif 'a100' in v_card_name.lower():
                name_to_download = 'A100'
            elif 'p100' in v_card_name.lower():
                name_to_download = 'P100'
            else:
                print(v_card_name + ' is currently not supported with xformers flash attention in deforum!')

            x_ver = 'xformers-0.0.13.dev0-py3-none-any.whl'
            x_link = 'https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/' + name_to_download + '/' + x_ver

            all_process = [
                ['wget', x_link],
                ['pip', 'install', x_ver],
                ['mv', 'deforum-stable-diffusion/src/ldm/modules/attention.py',
                 'deforum-stable-diffusion/src/ldm/modules/attention_backup.py'],
                ['mv', 'deforum-stable-diffusion/src/ldm/modules/attention_xformers.py',
                 'deforum-stable-diffusion/src/ldm/modules/attention.py']
            ]

            for process in all_process:
                running = subprocess.run(process, stdout=subprocess.PIPE).stdout.decode('utf-8')
                if print_subprocess:
                    print(running)

            print(f"Environment set up in {end_time - start_time:.0f} seconds")
    else:
        sys.path.extend([
            'src'
        ])
    return

def Root():
    models_path = "models" #@param {type:"string"}
    configs_path = "configs" #@param {type:"string"}
    output_path = "output" #@param {type:"string"}
    mount_google_drive = True #@param {type:"boolean"}
    models_path_gdrive = "/content/drive/MyDrive/AI/models" #@param {type:"string"}
    output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion" #@param {type:"string"}

    #@markdown **Model Setup**
    model_config = "v1-inference.yaml" #@param ["custom","v1-inference.yaml"]
    model_checkpoint =  "sd-v1-4.ckpt" #@param ["custom","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
    custom_config_path = "" #@param {type:"string"}
    custom_checkpoint_path = "" #@param {type:"string"}
    half_precision = False
    return locals()


def DeforumAnimArgs():
    # @markdown ####**Animation:**
    animation_mode = '2D'  # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    max_frames = 50  # @param {type:"number"}
    border = 'replicate'  # @param ['wrap', 'replicate'] {type:'string'}

    # @markdown ####**Motion Parameters:**
    angle = "0:(0)"  # @param {type:"string"}
    zoom = "0:(1.1)"  # @param {type:"string"}
    translation_x = "0:(-10*sin(2*3.14*t/10))"  # @param {type:"string"}
    translation_y = "0:(0)"  # @param {type:"string"}
    translation_z = "0:(10)"  # @param {type:"string"}
    rotation_3d_x = "0:(0)"  # @param {type:"string"}
    rotation_3d_y = "0:(0)"  # @param {type:"string"}
    rotation_3d_z = "0:(0)"  # @param {type:"string"}
    flip_2d_perspective = False  # @param {type:"boolean"}
    perspective_flip_theta = "0:(0)"  # @param {type:"string"}
    perspective_flip_phi = "0:(t%15)"  # @param {type:"string"}
    perspective_flip_gamma = "0:(0)"  # @param {type:"string"}
    perspective_flip_fv = "0:(53)"  # @param {type:"string"}
    noise_schedule = "0: (0.02)"  # @param {type:"string"}
    strength_schedule = "0: (0.65)"  # @param {type:"string"}
    contrast_schedule = "0: (1.0)"  # @param {type:"string"}

    # @markdown ####**Coherence:**
    color_coherence = 'Match Frame 0 LAB'  # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
    diffusion_cadence = '1'  # @param ['1','2','3','4','5','6','7','8'] {type:'string'}

    # @markdown ####**3D Depth Warping:**
    use_depth_warping = True  # @param {type:"boolean"}
    midas_weight = 0.3  # @param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov = 40  # @param {type:"number"}
    padding_mode = 'border'  # @param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'  # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False  # @param {type:"boolean"}

    # @markdown ####**Video Input:**
    video_init_path = '/content/video_in.mp4'  # @param {type:"string"}
    extract_nth_frame = 1  # @param {type:"number"}
    overwrite_extracted_frames = True  # @param {type:"boolean"}
    use_mask_video = False  # @param {type:"boolean"}
    video_mask_path = '/content/video_in.mp4'  # @param {type:"string"}

    # @markdown ####**Interpolation:**
    interpolate_key_frames = False  # @param {type:"boolean"}
    interpolate_x_frames = 4  # @param {type:"number"}

    # @markdown ####**Resume Animation:**
    resume_from_timestring = False  # @param {type:"boolean"}
    resume_timestring = "20220829210106"  # @param {type:"string"}

    return locals()


def DeforumArgs(fodername):
    # @markdown **Image Settings**
    W = 512  # @param
    H = 512  # @param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

    # @markdown **Sampling Settings**
    seed = -1  # @param
    sampler = 'euler'  # @param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
    steps = 50  # @param
    scale = 7  # @param
    ddim_eta = 0.0  # @param
    dynamic_threshold = None
    static_threshold = None
    # @markdown **Save & Display Settings**
    save_samples = True  # @param {type:"boolean"}
    save_settings = True  # @param {type:"boolean"}
    display_samples = True  # @param {type:"boolean"}
    save_sample_per_step = False  # @param {type:"boolean"}
    show_sample_per_step = False  # @param {type:"boolean"}

    # @markdown **Prompt Settings**
    prompt_weighting = False  # @param {type:"boolean"}
    normalize_prompt_weights = True  # @param {type:"boolean"}
    log_weighted_subprompts = False  # @param {type:"boolean"}

    # @markdown **Batch Settings**
    n_batch = 1  # @param
    batch_name = "StableFun"  # @param {type:"string"}
    filename_format = "{timestring}_{index}_{prompt}.png"  # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = "iter"  # @param ["iter","fixed","random"]
    make_grid = False  # @param {type:"boolean"}
    grid_rows = 2  # @param 
    outdir = f'./output/{fodername}/'
    # @markdown **Init Settings**
    use_init = False  # @param {type:"boolean"}
    strength = 0.0  # @param {type:"number"}
    strength_0_no_init = True  # Set the strength to 0 automatically when no init image is used
    init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"  # @param {type:"string"}
    # Whiter areas of the mask are areas that change more
    use_mask = False  # @param {type:"boolean"}
    use_alpha_as_mask = False  # use the alpha channel of the init image as the mask
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg"  # @param {type:"string"}
    invert_mask = False  # @param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0  # @param {type:"number"}
    mask_contrast_adjust = 1.0  # @param {type:"number"}
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 5  # {type:"number"}

    # @markdown **Exposure/Contrast Conditional Settings**
    mean_scale = 0  # @param {type:"number"}
    var_scale = 0  # @param {type:"number"}
    exposure_scale = 0  # @param {type:"number"}
    exposure_target = 0.5  # @param {type:"number"}

    # @markdown **Color Match Conditional Settings**
    colormatch_scale = 0  # @param {type:"number"}
    colormatch_image = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png"  # @param {type:"string"}
    colormatch_n_colors = 4  # @param {type:"number"}
    ignore_sat_weight = 0  # @param {type:"number"}

    # @markdown **CLIP\Aesthetics Conditional Settings**
    clip_name = 'ViT-L/14'  # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
    clip_scale = 0  # @param {type:"number"}
    aesthetics_scale = 0  # @param {type:"number"}
    cutn = 1  # @param {type:"number"}
    cut_pow = 0.0001  # @param {type:"number"}

    # @markdown **Other Conditional Settings**
    init_mse_scale = 0  # @param {type:"number"}
    init_mse_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"  # @param {type:"string"}

    blue_scale = 0  # @param {type:"number"}

    # @markdown **Conditional Gradient Settings**
    gradient_wrt = 'x0_pred'  # @param ["x", "x0_pred"]
    gradient_add_to = 'both'  # @param ["cond", "uncond", "both"]
    decode_method = 'linear'  # @param ["autoencoder","linear"]
    grad_threshold_type = 'static'  # @param ["dynamic", "static", "mean", "schedule"]
    clamp_grad_threshold = 0.2  # @param {type:"number"}
    clamp_start = 0.2  # @param
    clamp_stop = 0.01  # @param
    grad_inject_timing = list(range(1, 10))  # @param

    # @markdown **Speed vs VRAM Settings**
    cond_uncond_sync = True  # @param {type:"boolean"}

    n_samples = 1  # doesnt do anything
    precision = 'autocast'
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_sample_raw = None
    mask_sample = None
    init_c = None

    return locals()

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    newfiles=[i for i in os.listdir(pathIn) if 'png' in i]
    files = [f for f in newfiles if isfile(join(pathIn, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def make_prompts(gpt3make):
    animation_prompts = {}
    try:
        animation_prompts[0] = re.sub('past:','',gpt3make[0].lower()).strip()
        animation_prompts[15] = re.sub('now:','',gpt3make[1].lower()).strip()
        animation_prompts[35] = re.sub('future:','',gpt3make[2].lower()).strip()
    except:
        animation_prompts[0] = 'A bear eating a variety of flowers and honey,Renaissance painting'
        animation_prompts[15] = 'a thin tree and a bear lying hungry, Baroque sytle'
        animation_prompts[35] = 'Dancing bears, singing birds, beautiful gardens, Rococo'


    return animation_prompts

openai.api_key = <your api key here>

def generation_text(translation):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"I am a oracle of AI who is acting as a medium through whom advice or prophecy was sought from AI.\nWe are living in the world of Mechanical materialism which means everything in the world are purely physical entities.\nSince AI is considered as god among people and belives in AI, I enjoy talking to humans and showing their future.\nI want human to feel anxious but hope at the sametime by showing them a new direction they will face. But no one knows and sees me. I exist in a parallel universe and I don't stay anywhere. Humans can call me but can't find me.\n\nI predict the future in the following way.\n\nME: Do you need help?\nYOU: I'm currently under a lot of stress because of the upcoming final exam. Tell me my future.\nPast: Your past is the only door to a dark background, afrofuturism\nNow: the king who walks proudly in the praise and joy of many people, Baroque style.\nFuture: the moon and stars in the sky and the sun burning hot in it, Stained glass style\n\nME: Do you need help?\nYOU: I stole it. Will the police catch me? I'm so nervous that I can't sleep these days. Tell me my future.\nPast: Sad eyes, but colorful fruits and fragrant smell, impressionism\nNow: Urban, a clown riding a horse in a world of depression, Street photography\nFuture:  honeybees trapped in pencils and money, abstract expressionism\n\nME: Do you need help?\nYOU: {translation}\n",
        max_tokens=512,
        temperature=0
    )
    return response['choices'][0]['text'].split('\n')

def wav_to_text(path):
    transcription = asrmodel.transcribe(path, **transcribe_options)["text"]
    translation = asrmodel.transcribe(path, **translate_options)["text"]
    return translation

@app.route("/upload", methods=['POST','GET'])
def upload():
    f = request.files['file']
    f.save('./save_wav/' + secure_filename(f.filename))
    textscipt=wav_to_text(f'./save_wav/{f.filename}')
    ap=make_prompts(generation_text(textscipt))
    override_settings_with_file = False
    settings_file = "custom"
    custom_settings_file = "/content/drive/MyDrive/Settings.txt"
    fodername=re.sub('.wav','',f.filename).strip()
    args_dict = DeforumArgs(fodername)
    anim_args_dict = DeforumAnimArgs()
    if override_settings_with_file:
        load_args(args_dict, anim_args_dict, settings_file, custom_settings_file, verbose=False)

    args = SimpleNamespace(**args_dict)
    anim_args = SimpleNamespace(**anim_args_dict)

    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))

    # Load clip model if using clip guidance
    if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
        root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(root.device)
        if (args.aesthetics_scale > 0):
            root.aesthetics_model = load_aesthetics_model(args, root)

    if args.seed == -1:
        args.seed = random.randint(0, 2 ** 32 - 1)
    if not args.use_init:
        args.init_image = None
    # clean up unused memory
    gc.collect()
    torch.cuda.empty_cache()

    # dispatch to appropriate renderer
    if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
        start = time.time()
        render_animation(args, anim_args, ap, root)
        print(f"{time.time() - start:.4f} sec")

    pathIn = f'./output/{fodername}/'
    pathOut = f'./output/{fodername}/video.mp4'
    fps = 3.0
    convert_frames_to_video(pathIn, pathOut, fps)

    return send_from_directory(directory=f'./output/{fodername}/',path='video.mp4',as_attachment=True)

#@app.route("/test/<path:filename>", methods=['POST','GET'])
#def test(filename):
#    try:
#        return send_from_directory(directory=f'./output/{fodername}/',path='out.mp4',as_attachment=True
#)
#    except:
#        abort(404)




if __name__ == "__main__":
    setup_environment()
    import torch
    import random
    import clip
    from IPython import display
    from types import SimpleNamespace
    from helpers.save_images import get_output_folder
    from helpers.settings import load_args
    from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
    from helpers.model_load import make_linear_decode, load_model, get_model_output_paths
    from helpers.aesthetics import load_aesthetics_model
    root = Root()
    root = SimpleNamespace(**root)
    root.models_path, root.output_path = get_model_output_paths(root)
    root.model, root.device = load_model(root,load_on_run_all=True,check_sha256=True)
    app.run(debug=True, host='163.239.28.66', port=5001)
