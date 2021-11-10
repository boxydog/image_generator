import math
import io
from pathlib import Path
from re import sub
import time
import traceback

from omegaconf import OmegaConf
from PIL import Image
import requests
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm, trange
import kornia.augmentation as K

from clip import clip

import warnings

warnings.simplefilter("ignore")  # Avoid spookin people for Cumin


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x),
                       x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic',
                         align_corners=align_corners)


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(
        dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (
                    input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(
            2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists,
                                                                     self.stop)).mean()


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith(
            'https://'):
        headers = {
            "user-agent": "AIArtMachineBot/0.0 (https://is.gd/aiartmachine; h@hillelwayne.com) generic-library/0.0"}
        r = requests.get(url_or_path, headers=headers)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def save_url_to_file(url, file):
    """save content of URL to file."""
    # TODO(dan): Factor out with 'fetch' above
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Couldn't load {url}")
    with open(file, "wb") as file:
        file.write(response.content)


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 3)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = [prompt]
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean()


class MakeCutoutsDefault(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (
                        max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


class MakeCutoutsCumin(nn.Module):
    """from https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ"""

    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7,
                           padding_mode='border'),
            K.RandomPerspective(0.7, p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1 / .3), same_on_batch=True, p=0.7),

        )
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []

        for _ in range(self.cutn):
            cutout = (self.av_pool(input) + self.max_pool(input)) / 2
            cutouts.append(cutout)
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(
                0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(
            **config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.encoder, model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
    return image.resize(size, Image.LANCZOS)


class ImageGenerator:
    def __init__(self, args, prompt, flavor, image_size,
                 reset=True, display_image=None):
        self.args = args
        self.prompt = prompt
        self.flavor = flavor
        self.image_size = image_size
        self.step_size = 0.05 * (args.weirdness if args.weirdness != 11 else 22)
        self.display_image = display_image

        # I hate using underscores for names,
        # but this way is better for people using utf-8
        self.prompt_filename = sub(
            "\W", "", self.prompt.lower().replace(" ", "_")) + '.png'

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        vqgan_config = 'vqgan_imagenet_f16_1024.yaml'
        vqgan_checkpoint = 'vqgan_imagenet_f16_1024.ckpt'

        # re-load the vqgan checkpoint
        ckpt_file = 'vqgan_imagenet_f16_1024.ckpt'
        if reset or not Path(ckpt_file).is_file():
            save_url_to_file(
                'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1',
                ckpt_file
            )

        self.model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(self.device)
        self.perceptor = clip.load("ViT-B/32", jit=False)[
            0].eval().requires_grad_(False).to(self.device)

        self.cut_size = self.perceptor.visual.input_resolution
        self.e_dim = self.model.quantize.e_dim
        self.f = 2 ** (self.model.decoder.num_resolutions - 1)
        self.flavordict = {
            "default": MakeCutoutsDefault,
            "cumin": MakeCutoutsCumin,
            "rosewater": MakeCutoutsDefault,
            "oregano": MakeCutoutsDefault,
            "thyme": MakeCutoutsCumin,
        }
        self.make_cutouts = self.flavordict[self.flavor](
            self.cut_size, self.args.cutn, cut_pow=self.args.cut_pow)
        self.n_toks = self.model.quantize.n_e
        self.toksX, self.toksY = self.image_size // self.f, self.image_size // self.f
        self.sideX, self.sideY = self.toksX * self.f, self.toksY * self.f

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)

        """
        Oregano and Thyme are based on z+quantize, from
        https://colab.research.google.com/drive/1wkF67ThUz37T2_oPIuSwuO4e_-0vjaLs
        They use a completely different means of guiding VQGAN vs codebook sampling.
        Best way to handle this rn is to branch on the logic. A smarter, less lazy
        person would have instead made Task objects with injectable Flavor objects, but I am
        neither smart nor less lazy.
        """
        if self.uses_zq():
            oh = F.one_hot(torch.randint(
                self.n_toks, [self.toksX * self.toksY], device=self.device),
                self.n_toks).float()
            z = oh @ self.model.quantize.embedding.weight
            z = z.view([-1, self.toksY, self.toksX, self.e_dim]).permute(
                0, 3, 1, 2)
            z = torch.rand_like(z) * 2
            z.requires_grad_(True)  # Does this slow down basic operations?
            self.z = z

            self.opt = optim.AdamW(
                [z], lr=self.step_size,
                weight_decay=self.args.weight_decay)
        else:
            self.logits = torch.randn([self.toksY * self.toksX, self.n_toks],
                                      device=self.device,
                                      requires_grad=True)

            self.opt = optim.AdamW(
                [self.logits], lr=self.step_size,
                weight_decay=self.args.weight_decay)

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])

        self.pMs = []

        # make the output path for dumping the images
        img_path = Path("img") / "grouped" / self.prompt_filename
        all_path = Path("img/all")
        img_path.mkdir(parents=True, exist_ok=True)
        all_path.mkdir(parents=True, exist_ok=True)

        for prompt in self.args.prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = self.perceptor.encode_text(
                clip.tokenize(txt).to(self.device)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(self.device))

        for prompt in self.args.image_prompts:
            path, weight, stop = parse_prompt(prompt)
            img = resize_image(Image.open(fetch(path)).convert('RGB'),
                               (self.sideX, self.sideY))
            batch = self.make_cutouts(TF.to_tensor(img)[None].to(self.device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(self.device))

        for seed, weight in zip(self.args.noise_prompt_seeds,
                                self.args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(
                generator=gen)
            self.pMs.append(Prompt(embed, weight).to(self.device))

    def uses_zq(self):
        return self.flavor in {"oregano", "thyme"}

    def z_synth(self):
        z_q = vector_quantize(
            self.z.movedim(1, 3),
            self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    def logit_synth(self, one_hot):
        z = one_hot @ self.model.quantize.embedding.weight
        z = z.view([-1, self.toksY, self.toksX, self.e_dim]).permute(0, 3, 1, 2)
        return clamp_with_grad(self.model.decode(z).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkin(self, i, losses):
        tqdm.write(f'iterations: {i}, prompt: {self.prompt}')
        if self.uses_zq():
            out = self.z_synth()
        else:
            one_hot = F.one_hot(
                self.logits.argmax(1), self.n_toks).to(self.logits.dtype)
            out = self.logit_synth(one_hot)
        out_img = TF.to_pil_image(out[0].cpu())
        out_img.save(self.prompt_filename)

        if self.display_image:
            self.display_image(self.prompt_filename)

    def ascend_txt(self):
        self.opt.zero_grad(set_to_none=True)
        if self.uses_zq():
            out = self.z_synth()
        else:
            probs = self.logits.softmax(1)
            one_hot = F.one_hot(probs.multinomial(1)[..., 0], self.n_toks).to(
                self.logits.dtype)
            one_hot = replace_grad(one_hot, probs)
            out = self.logit_synth(one_hot)
        iii = self.perceptor.encode_image(
            self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.args.tv_weight:
            result.append(tv_loss(out) * self.args.tv_weight / 4)

        for prompt in self.pMs:
            result.append(prompt(iii))

        return result

    def train(self, idx):
        self.opt.zero_grad(set_to_none=True)
        lossAll = self.ascend_txt()
        if idx % self.args.display_freq == 0:
            self.checkin(idx, lossAll)
        loss = sum(lossAll)
        loss.backward()
        self.opt.step()

    def generate_image(self):
        i = 1
        print("NOTE: First image will look random. This is normal.")
        try:
            with trange(self.args.total_iterations) as pbar:
                start = time.perf_counter()
                while i < self.args.total_iterations:
                    pbar.update()
                    self.train(i)
                    if i == 10:
                        end = time.perf_counter()
                        some_minutes = int(
                            (end - start) * self.args.total_iterations // (
                                        60 * 10))
                        print(
                            f"It will take about {some_minutes} minutes to complete all {self.args.total_iterations} iterations.")
                    i += 1
        except KeyboardInterrupt:
            pass
        except RuntimeError as err:
            print("ERROR! ERROR! ERROR!")
            print("Possibly the image size you chose was too big!")
            print(f"Error: {err}")
            traceback.print_exc()
        else:
            print("Final image.")
            self.checkin(i, 0)

        print("All done!")
