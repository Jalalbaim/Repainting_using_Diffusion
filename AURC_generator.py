import os
import argparse
import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageFilter
import random

import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)
from Evaluation_Method import SaliencyAttributor
from knockout import KnockoutMaskGenerator


PCT              = 0.30
MEANS            = [0.485, 0.456, 0.406]
STDS             = [0.229, 0.224, 0.225]
VGG_PATH         = "data/weights/vgg16-397923af.pth"
DEEPFILL_PATH    = "deepfillv2-pytorch/pretrained/states_pt_places2.pth"
OUTPUT_ROOT      = "./latest_outputs"
BLUR_RADIUS      = 31        # PIL GaussianBlur radius — large value
FOLDERS          = ["GT", "masked", "blurred", "repainted", "deepfill"]


def load_vgg(path):
    model = models.vgg16(weights=None)
    model.load_state_dict(th.load(path, map_location="cpu"))
    model.eval()
    return model


def load_deepfill(checkpoint_path: str, device):
    import sys
    _deepfill_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepfillv2-pytorch")
    if _deepfill_repo not in sys.path:
        sys.path.append(_deepfill_repo)

    raw = th.load(checkpoint_path, map_location="cpu")
    state = raw["G"] if "G" in raw else raw

    if "stage1.conv1.conv.weight" in state:
        from model.networks import Generator
        generator = Generator(cnum_in=5, cnum_out=3, cnum=48, return_flow=False)
    else:
        from model.networks_tf import Generator
        generator = Generator(cnum_in=5, cnum=48, return_flow=False)
    generator.load_state_dict(state)
    generator.to(device).eval()
    return generator


def deepfill_inpaint(generator, img_np: np.ndarray, mask_np: np.ndarray, device) -> np.ndarray:
    H, W = img_np.shape[:2]
    H8, W8 = H // 8 * 8, W // 8 * 8

    img_t = (th.from_numpy(img_np[:H8, :W8]).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0).to(device)

    df_mask = th.from_numpy((1.0 - mask_np[:H8, :W8]).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    img_masked = img_t * (1.0 - df_mask)
    ones = th.ones_like(df_mask)
    x = th.cat([img_masked, ones, ones * df_mask], dim=1) 

    with th.inference_mode():
        _, x_stage2 = generator(x, df_mask)

    result_t = img_t * (1.0 - df_mask) + x_stage2 * df_mask
    result_crop = ((result_t.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    if H8 == H and W8 == W:
        return result_crop
    out = img_np.copy()
    out[:H8, :W8] = result_crop
    return out


def to_uint8(t):
    """Diffusion tensor (B,C,H,W) in [-1,1] → numpy (H,W,3) uint8."""
    t = ((t + 1) * 127.5).clamp(0, 255).to(th.uint8)
    return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def make_output_dirs(root):
    for folder in FOLDERS:
        os.makedirs(os.path.join(root, folder), exist_ok=True)


def build_variants(img_np: np.ndarray, method_xai: str, k: int):

    img_pil = Image.fromarray(img_np)
    prep = transforms.Compose([
        transforms.Resize(img_np.shape[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEANS, std=STDS),
    ])
    inp = prep(img_pil).unsqueeze(0)

    attributor = SaliencyAttributor(model=load_vgg(VGG_PATH), method=method_xai)
    attr = attributor.compute(inp)

    masker = KnockoutMaskGenerator(K=k, means=MEANS, stds=STDS)
    out_knock, mask, _ = masker.generate(inp, attr)

    masked_np = (out_knock.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)

    mask_np = mask.numpy()
    blurred_full = np.array(img_pil.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS)))
    blurred_np = np.where(mask_np[:, :, None] == 0, blurred_full, img_np).astype(np.uint8)

    return masked_np, blurred_np, mask


def main(conf: conf_mgt.Default_Conf):
    print("Start AURC Generator")

    seed = conf.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("Device:", device)

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print("Diffusion model loaded")

    # guidance classifier
    cond_fn = None
    if conf.classifier_scale > 0 and conf.classifier_path:
        classifier = create_classifier(**select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(conf.classifier_path), map_location="cpu")
        )
        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, **_):
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                logp = F.log_softmax(logits, dim=-1)
                sel = logp[range(len(logits)), y.view(-1)]
                return th.autograd.grad(sel.sum(), x_in)[0] * conf.classifier_scale

        print("Guidance classifier loaded")

    def model_fn(x, t, y=None, gt=None, **_):
        return model(x, t, y if conf.class_cond else None, gt=gt)

    # DeepFill v2 model
    deepfill = load_deepfill(DEEPFILL_PATH, device)
    print("DeepFill v2 model loaded")

    eval_name = conf.get_default_eval_name()
    dl = conf.get_dataloader(dset="eval", dsName=eval_name)

    method_xai = conf.get("method_xai", "saliency")
    print(f"XAI method : {method_xai}")

    make_output_dirs(OUTPUT_ROOT)

    k = int(conf.image_size * conf.image_size * PCT)
    print(f"Deletion rate: {int(PCT * 100)}%  →  K={k} pixels\n")

    # main loop 
    for batch in dl:
        for key, val in batch.items():
            if isinstance(val, th.Tensor):
                batch[key] = val.to(device)

        gt      = batch["GT"]
        img_name = batch["GT_name"][0].split(".")[0]
        print(f"Processing: {img_name}")

        img_np = to_uint8(gt)[0]

        # GT
        Image.fromarray(img_np).save(os.path.join(OUTPUT_ROOT, "GT", f"{img_name}.png"))

        # Masked & Blurred
        masked_np, blurred_np, mask = build_variants(img_np, method_xai, k)
        Image.fromarray(masked_np).save(os.path.join(OUTPUT_ROOT, "masked",  f"{img_name}.png"))
        Image.fromarray(blurred_np).save(os.path.join(OUTPUT_ROOT, "blurred", f"{img_name}.png"))

        # Repainted
        mask_np = mask.numpy()
        mask_4d = th.from_numpy(mask_np).float().to(device).unsqueeze(0).unsqueeze(0)

        y_classes = (
            th.ones(gt.size(0), dtype=th.long, device=device) * conf.cond_y
            if conf.cond_y is not None
            else th.randint(0, NUM_CLASSES, (gt.size(0),), device=device)
        )

        sample_fn = diffusion.ddim_sample_loop if conf.use_ddim else diffusion.p_sample_loop
        result = sample_fn(
            model_fn,
            (gt.size(0), 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs={"gt": gt, "gt_keep_mask": mask_4d, "y": y_classes},
            cond_fn=cond_fn,
            device=device,
            progress=conf.show_progress,
            return_all=True,
            conf=conf,
        )
        repainted_np = to_uint8(result["sample"])[0]
        Image.fromarray(repainted_np).save(os.path.join(OUTPUT_ROOT, "repainted", f"{img_name}.png"))

        # DeepFill v2
        deepfill_np = deepfill_inpaint(deepfill, img_np, mask_np, device)
        Image.fromarray(deepfill_np).save(os.path.join(OUTPUT_ROOT, "deepfill", f"{img_name}.png"))

        print(f"  Saved: GT / masked / blurred / repainted / deepfill  {OUTPUT_ROOT}/\n")

    print(f"Generator done. All outputs in {OUTPUT_ROOT}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    conf = conf_mgt.conf_base.Default_Conf()
    conf.update(yamlread(args.conf_path))
    main(conf)
