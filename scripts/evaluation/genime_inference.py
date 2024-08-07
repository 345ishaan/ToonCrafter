import argparse, os, sys, glob
import datetime, time
from types import SimpleNamespace
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict
from io import BytesIO
import requests
import numpy as np
import cv2
import torch
import uuid
import json
import io
from zipfile import ZipFile
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config


def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

def load_data_prompts(data_dir, video_size=(256,256), video_frames=16, interp=False):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    ## load prompts
    prompt_file = get_filelist(data_dir, ['txt'])
    assert len(prompt_file) > 0, "Error: found NO prompt file!"
    ###### default prompt
    default_idx = 0
    default_idx = min(default_idx, len(prompt_file)-1)
    if len(prompt_file) > 1:
        print(f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
    ## only use the first one (sorted by name) if multiple exist
    
    ## load video
    file_list = get_filelist(data_dir, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
    # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
    data_list = []
    filename_list = []
    prompt_list = load_prompts(prompt_file[default_idx])
    n_samples = len(prompt_list)
    for idx in range(n_samples):
        if interp:
            image1 = Image.open(file_list[2*idx]).convert('RGB')
            image_tensor1 = transform(image1).unsqueeze(1).flip(1) # [c,1,h,w]
            image2 = Image.open(file_list[2*idx+1]).convert('RGB')
            image_tensor2 = transform(image2).unsqueeze(1).flip(1) # [c,1,h,w]
            frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
            _, filename = os.path.split(file_list[idx*2])
        else:
            image = Image.open(file_list[idx]).convert('RGB')
            image_tensor = transform(image).unsqueeze(1) # [c,1,h,w]
            frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
            _, filename = os.path.split(file_list[idx])

        data_list.append(frame_tensor)
        filename_list.append(filename)
        
    return filename_list, data_list, prompt_list


def load_data_prompts_genime(img_urls, prompts, cache_map, save_img_dir, video_size=(256,256), video_frames=16, interp=False):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    assert len(img_urls) == len(prompts)

    def load_img_from_url(img_url):
        if cache_map.get(img_url, None):
            return Image.open(cache_map.get(img_url)).convert('RGB')
        response = requests.get(img_url)
        if response.status_code == 200:
            image_data = BytesIO(response.content)
            face_img = Image.open(image_data).convert('RGB')
            #image_data = np.asarray(bytearray(response.content), dtype="uint8")
            #face_img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            #assert face_img is not None
            #face_img = Image.fromarray(face_img)
            return face_img
        raise Exception("Unable to load URL")
    
    def _update_cache(url, data):
        if url in cache_map:
            return
        if not os.path.exists(save_img_dir):
            os.makedirs(save_img_dir)
        fname = os.path.join(save_img_dir, str(uuid.uuid4()) +".png")
        data.save(fname)
        cache_map[url] = fname
        
    
    data_list = []
    filename_list = []
    prompt_list = [str(p) for p in prompts]
    n_samples = len(prompt_list)
    import pdb

    for idx in range(n_samples):
        if interp:
            img_url_a, img_url_b = img_urls[idx]
            image1 = load_img_from_url(img_url_a)
            _update_cache(img_url_a, image1)
            image_tensor1 = transform(image1).unsqueeze(1) # [c,1,h,w]
            image2 = load_img_from_url(img_url_b)
            _update_cache(img_url_a, image1)
            image_tensor2 = transform(image2).unsqueeze(1) # [c,1,h,w]
            frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
            frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)

            fname = cache_map[img_url_a]
            _, filename = os.path.split(fname)
        else:
            image = load_img_from_url(img_urls[idx])
            _update_cache(img_urls[idx], image)
            image_tensor = transform(image).unsqueeze(1) # [c,1,h,w]
            frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
            fpath = cache_map[img_urls[idx]]
            _, filename = os.path.split(fpath)

        data_list.append(frame_tensor)
        filename_list.append(filename)
    return filename_list, data_list, prompt_list


def save_results(prompt, samples, filename, fakedir, fps=8, loop=False):
    filename = filename.split('.')[0]+'.mp4'
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        if loop:
            video = video[:-1,...]
        
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, h, n*w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        path = os.path.join(savedirs[idx], filename)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'}) ## crf indicates the quality
        zip_filename = path.replace('mp4', 'zip')
        with ZipFile(zip_filename, 'w') as zipf:
            for i in range(grid.shape[0]):
                img = Image.fromarray(grid[i].cpu().numpy())
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                zipf.writestr(f'frame_{i:04d}.png', buffer.read())


def zip_video_frames(video_path, zip_fpath):
    video, audio, info = torchvision.io.read_video(video_path)
    fps = info['video_fps']
    frame_count = video.shape[0]
    with ZipFile(zip_fpath, 'w') as zipf:
        for frame_number, frame in enumerate(video):
            pil_image = torchvision.transforms.ToPILImage()(frame)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            buffer.seek(0)
            zipf.writestr(f'vid_fram_{frame_number:04d}.png', buffer.read())
    return fps, frame_count



def save_results_seperate(prompt, samples, filename, fakedir, fps=10, loop=False):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        if loop: # remove the last frame
            video = video[:,:,:-1,...]
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        assert n == 1
        for i in range(n):
            grid = video[i,...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0) #thwc
            path = os.path.join(savedirs[idx], 'final_concat.mp4')
            torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})

            zip_filename = os.path.join(savedirs[idx], filename.replace('png', 'zip'))
            zip_filename = path.replace('mp4', 'zip')

            with ZipFile(zip_filename, 'w') as zipf:
                for i in range(grid.shape[0]):
                    img = Image.fromarray(grid[i].cpu().numpy())
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    buffer.seek(0)
                    zipf.writestr(f'frame_{i:04d}.png', buffer.read())


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z

def get_latent_z_with_hidden_states(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    encoder_posterior, hidden_states = model.first_stage_model.encode(x, return_hidden_states=True)

    hidden_states_first_last = []
    ### use only the first and last hidden states
    for hid in hidden_states:
        hid = rearrange(hid, '(b t) c h w -> b c t h w', t=t)
        hid_new = torch.cat([hid[:, :, 0:1], hid[:, :, -1:]], dim=2)
        hidden_states_first_last.append(hid_new)

    z = model.get_first_stage_encoding(encoder_posterior).detach()
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z, hidden_states_first_last

def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""]*batch_size

    img = videos[:,:,0] #bchw
    img_emb = model.embedder(img) ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z, hs = get_latent_z_with_hidden_states(model, videos) # b c t h w
        if loop or interp:
            img_cat_cond = torch.zeros_like(z)
            img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
            img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
        else:
            img_cat_cond = z[:,:,:1,:,:]
            img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond] # b c 1 h w
    
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    additional_decode_kwargs = {'ref_context': hs}

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:

            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            mask=cond_mask,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples, **additional_decode_kwargs)

        index = list(range(samples.shape[2]))
        del index[1]
        del index[-2]
        samples = samples[:,:,index,:,:]
        ## reconstruct from latent to pixel space
        batch_images_middle = model.decode_first_stage(samples, **additional_decode_kwargs)
        batch_images[:,:,batch_images.shape[2]//2-1:batch_images.shape[2]//2+1] = batch_images_middle[:,:,batch_images.shape[2]//2-2:batch_images.shape[2]//2]



        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)



class InterPolater:

    def __init__(self) -> None:
        self.args = SimpleNamespace(
            savedir=None,
            ckpt_path="checkpoints/tooncrafter_512_interp_v1/model.ckpt",
            config="configs/inference_512_v1.0.yaml",
            prompt_dir="prompts/512_interp_small/",
            n_samples=1,
            ddim_steps=50,
            ddim_eta=1.0,
            bs=1,
            height=320,
            width=512,
            frame_stride=10,
            unconditional_guidance_scale=7.5,
            seed=123,
            video_length=16,
            negative_prompt=False,
            text_input=True,
            multiple_cond_cfg=False,
            cfg_img=None,
            timestep_spacing="uniform_trailing",
            guidance_rescale=0.7,
            perframe_ae=True,
            loop=False,
            interp=True,
            cache_fname="./cache.json",
            save_img_dir="./save_img_dir"
        )
        
        seed_everything(self.args.seed)
        self.config = OmegaConf.load(self.args.config)
        self.model_config = self.config.pop("model", OmegaConf.create())

        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        self.model_config['params']['unet_config']['params']['use_checkpoint'] = False
        self.model = instantiate_from_config(self.model_config)
        self.model = self.model.cuda(0)
        self.model.perframe_ae = self.args.perframe_ae
        assert os.path.exists(self.args.ckpt_path), "Error: checkpoint Not Found!"
        self.model = load_model_checkpoint(self.model, self.args.ckpt_path)
        self.model.eval()

        assert self.args.bs == 1, "Current implementation only support [batch size = 1]!"
    
    def infer(self, img_urls, prompts, save_dir, eta=None, cfg_scale=None, steps=None, 
              video_len=None, frame_stride=None, fps=None, height=None, width=None):
        ## latent noise shape
        use_h = self.args.height if height is None else height
        use_w = self.args.width if width is None else width
        assert (use_h % 16 == 0) and (use_w % 16 == 0), "Error: image size [h,w] should be multiples of 16!"

        h, w = use_h // 8, use_w // 8
        channels = self.model.model.diffusion_model.out_channels
        n_frames = self.args.video_length if video_len is None else video_len
        print(f'Inference with {n_frames} frames')
        noise_shape = [self.args.bs, channels, n_frames, h, w]

        fakedir = save_dir
        fakedir_separate = save_dir

        # os.makedirs(fakedir, exist_ok=True)
        os.makedirs(fakedir_separate, exist_ok=True)

        ## prompt file setting
        assert self.args.save_img_dir != save_dir
        assert self.args.cache_fname and self.args.save_img_dir
        if not os.path.exists(self.args.save_img_dir):
            os.makedirs(self.args.save_img_dir)
        if not os.path.exists(self.args.cache_fname):
            cache_map = {}
            json.dump(cache_map, open(self.args.cache_fname, 'w'))
        else:
            cache_map = json.loads(open(self.args.cache_fname).read())

        
        assert os.path.exists(self.args.prompt_dir), "Error: prompt file Not Found!"
        filename_list, data_list, prompt_list = load_data_prompts_genime(img_urls, prompts, cache_map, self.args.save_img_dir,
                                                                         video_size=(use_h, use_w), 
                                                                         video_frames=n_frames, interp=self.args.interp)
        json.dump(cache_map, open(self.args.cache_fname, 'w'))
        

        num_samples = len(prompt_list)
        #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
        indices = list(range(0, num_samples))
        prompt_list_rank = [prompt_list[i] for i in indices]
        data_list_rank = [data_list[i] for i in indices]
        filename_list_rank = [filename_list[i] for i in indices]
        bs = self.args.bs
        start = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast():
            for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), bs)), desc='Sample Batch'):
                prompts = prompt_list_rank[indice:indice+bs]
                videos = data_list_rank[indice:indice+bs]
                filenames = filename_list_rank[indice:indice+bs]
                if isinstance(videos, list):
                    videos = torch.stack(videos, dim=0).to("cuda")
                else:
                    videos = videos.unsqueeze(0).to("cuda")

                batch_samples = image_guided_synthesis(self.model, prompts, videos, noise_shape, 
                                                       self.args.n_samples, 
                                                       self.args.ddim_steps if steps is None else steps, 
                                                       self.args.ddim_eta if eta is None else eta,
                                                       self.args.unconditional_guidance_scale if cfg_scale is None else cfg_scale, 
                                                       self.args.cfg_img, 
                                                       self.args.frame_stride if frame_stride is None else frame_stride, 
                                                       self.args.text_input, 
                                                       self.args.multiple_cond_cfg, self.args.loop, 
                                                       self.args.interp, self.args.timestep_spacing, self.args.guidance_rescale)

                ## save each example individually
                for nn, samples in enumerate(batch_samples):
                    ## samples : [n_samples,c,t,h,w]
                    prompt = prompts[nn]
                    filename = filenames[nn]
                    # save_results(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
                    save_results_seperate(prompt, samples, filename, fakedir, fps=8 if fps is None else fps, 
                                          loop=self.args.loop)

        print(f"Saved in {self.args.savedir}. Time used: {(time.time() - start):.2f} seconds")
        

        
def run_inference(args, gpu_num, gpu_no):
    ## model config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]

    fakedir = os.path.join(args.savedir, "samples")
    fakedir_separate = os.path.join(args.savedir, "samples_separate")

    # os.makedirs(fakedir, exist_ok=True)
    os.makedirs(fakedir_separate, exist_ok=True)

    ## prompt file setting
    assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
    filename_list, data_list, prompt_list = load_data_prompts(args.prompt_dir, video_size=(args.height, args.width), video_frames=n_frames, interp=args.interp)
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
    #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    prompt_list_rank = [prompt_list[i] for i in indices]
    data_list_rank = [data_list[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]

    start = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
            prompts = prompt_list_rank[indice:indice+args.bs]
            videos = data_list_rank[indice:indice+args.bs]
            filenames = filename_list_rank[indice:indice+args.bs]
            if isinstance(videos, list):
                videos = torch.stack(videos, dim=0).to("cuda")
            else:
                videos = videos.unsqueeze(0).to("cuda")

            batch_samples = image_guided_synthesis(model, prompts, videos, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                args.unconditional_guidance_scale, args.cfg_img, args.frame_stride, args.text_input, args.multiple_cond_cfg, args.loop, args.interp, args.timestep_spacing, args.guidance_rescale)

            ## save each example individually
            for nn, samples in enumerate(batch_samples):
                ## samples : [n_samples,c,t,h,w]
                prompt = prompts[nn]
                filename = filenames[nn]
                # save_results(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
                save_results_seperate(prompt, samples, filename, fakedir, fps=8, loop=args.loop)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/tooncrafter_512_interp_v1/model.ckpt", help="checkpoint path")
    parser.add_argument("--config", type=str, default="configs/inference_512_v1.0.yaml", help="config (yaml) path")
    parser.add_argument("--prompt_dir", type=str, default="prompts/512_interp_small/", help="a data dir containing videos and prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=320, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=10, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=True, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform_trailing", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.7, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=True, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")

    ## currently not support looping video and generative frame interpolation
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    parser.add_argument("--interp", action='store_true', default=True, help="generate generative frame interpolation or not")
    parser.add_argument("--cache_fname", type=str, default="./cache.json")
    parser.add_argument("--save_img_dir", type=str, default="./save_img_dir")
    return parser



if __name__ == "__main__":
    interpolater = InterPolater()
    # List of tuples where each tuple stores the start and end scene.
    img_urls = [
        ("https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/character_sam.webp?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2NoYXJhY3Rlcl9zYW0ud2VicCIsImlhdCI6MTcyMTQ3MDgzNSwiZXhwIjoxNzUzMDA2ODM1fQ.GXTUB7iYGkrEQIDJahtkdLFyInyetHgfSv5hgHPtvSk&t=2024-07-20T10%3A20%3A35.248Z",
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/character_sandy.webp?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2NoYXJhY3Rlcl9zYW5keS53ZWJwIiwiaWF0IjoxNzIxNDcyMDc5LCJleHAiOjE3NTMwMDgwNzl9.OnOoigkl4CJj8wsdZIDokwsRL4YK84o6o-O6A2plhYM&t=2024-07-20T10%3A41%3A18.610Z")
    ]
    img_urls = [
       ("https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/hanumanji.jpg?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2hhbnVtYW5qaS5qcGciLCJpYXQiOjE3MjIzMjU2NjAsImV4cCI6MTc1Mzg2MTY2MH0.jQVRaoHwPhvWOXdozEhAQFdCwskQeNxmkVqFXiXMkZA&t=2024-07-30T07%3A47%3A40.905Z",
        "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/hanumanji_2.jpg?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0L2hhbnVtYW5qaV8yLmpwZyIsImlhdCI6MTcyMjMyNTY3MywiZXhwIjoxNzUzODYxNjczfQ.DU9_IuZn_lC_83B6EGwcZnl074qo8LyuoVAmMWecmXY&t=2024-07-30T07%3A47%3A53.686Z")
    ]
    #img_urls = [
    #    ("https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/74906_1462_frame1.png?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0Lzc0OTA2XzE0NjJfZnJhbWUxLnBuZyIsImlhdCI6MTcyMjQwNjkzNCwiZXhwIjoxNzUzOTQyOTM0fQ.i3Df94vl1wy52vjlmiVcgV_Ft-4Mt2EiGQTJqQaBA3g&t=2024-07-31T06%3A22%3A14.885Z",
    #    "https://ttvaarlnqssopdguetwq.supabase.co/storage/v1/object/sign/genime-bucket/74906_1462_frame3.png?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJnZW5pbWUtYnVja2V0Lzc0OTA2XzE0NjJfZnJhbWUzLnBuZyIsImlhdCI6MTcyMjQwNjk2MiwiZXhwIjoxNzUzOTQyOTYyfQ.PR_1Zmql7DTxwwegLo35c19PfNOOVS6k-ACIxuXxTEw&t=2024-07-31T06%3A22%3A42.292Z")
    #]
    # List of prompts corresponding to the scene.
    prompt = [
        "man tearing his chest to reveal a divine image inside",
        #"walking man"
    ]
    save_dir = "/home/ToonCrafter/tooncrafter_results"
    interpolater.infer(img_urls, prompt, save_dir)
