"""
Single-image AURC & AUIC visualization at 30% masked pixels.
Compares RePaint and DeepFillV2 inpainting side-by-side.

Outputs saved to eagle_results/:
  GT.png, masked_aurc.png, masked_auic.png,
  repaint_aurc.png, repaint_auic.png,
  deepfillv2_aurc.png, deepfillv2_auic.png

@author: J.BAIM
"""

import os
import sys
import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

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

CONF_PATH      = "confs/test.yml"
VGG_PATH       = "data/weights/vgg16-397923af.pth"
DEEPFILL_CKPT  = "deepfillv2-pytorch/pretrained/states_tf_places2.pth"
OUTPUT_DIR     = "eagle_results"
MASK_PCT       = 0.30   # 30 % of pixels


def load_vgg16(path):
    model = models.vgg16(weights=None)
    model.load_state_dict(th.load(path, map_location="cpu"))
    model.eval()
    return model


def to_uint8(t):
    """[-1, 1] BCHW tensor → HWC uint8 numpy."""
    t = ((t + 1) * 127.5).clamp(0, 255).to(th.uint8)
    return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def build_masks(img_np, method_xai, pct, vgg_model):
    """
    Compute attribution and return two [1,1,H,W] float keep-masks (values 0/1):
      aurc_keep  – 1 = keep pixel,  0 = top-K important pixels to remove  (AURC)
      auic_keep  – 1 = top-K pixels to reveal, 0 = unimportant (to inpaint) (AUIC)
    """
    means = [0.485, 0.456, 0.406]
    stds  = [0.229, 0.224, 0.225]
    prep  = transforms.Compose([
        transforms.Resize(img_np.shape[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds),
    ])
    inp = prep(Image.fromarray(img_np)).unsqueeze(0)

    H, W = img_np.shape[:2]
    k    = int(H * W * pct)

    attr   = SaliencyAttributor(model=vgg_model, method=method_xai).compute(inp)
    _, mask, _ = KnockoutMaskGenerator(K=k, means=means, stds=stds).generate(inp, attr)
    # mask: [H,W] float, 0 at top-K important pixels, 1 elsewhere

    aurc_keep = mask.float().unsqueeze(0).unsqueeze(0)        # [1,1,H,W]
    auic_keep = (1.0 - mask).float().unsqueeze(0).unsqueeze(0)
    return aurc_keep, auic_keep


def make_masked_image(gt, keep_mask):
    """Overlay keep_mask on gt ([-1,1]); masked region → -1 (black after uint8)."""
    return gt * keep_mask + (-1.0) * (1.0 - keep_mask)


def inpaint_repaint(gt, keep_mask, model_fn, diffusion, conf, cond_fn, device):
    """RePaint inpainting.  keep_mask [1,1,H,W]: 1 = keep original, 0 = inpaint."""
    y_classes = (
        th.ones(gt.size(0), dtype=th.long, device=device) * conf.cond_y
        if conf.cond_y is not None
        else th.randint(0, NUM_CLASSES, (gt.size(0),), device=device)
    )
    model_kwargs = {"gt": gt, "gt_keep_mask": keep_mask, "y": y_classes}
    sample_fn = diffusion.ddim_sample_loop if conf.use_ddim else diffusion.p_sample_loop
    result = sample_fn(
        model_fn,
        (gt.size(0), 3, conf.image_size, conf.image_size),
        clip_denoised=conf.clip_denoised,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        device=device,
        progress=conf.show_progress,
        return_all=True,
        conf=conf,
    )
    return result["sample"]   # [-1,1] BCHW


def load_deepfillv2(ckpt_path, device):
    dfv2_path = os.path.abspath("deepfillv2-pytorch")
    sys.path.insert(0, dfv2_path)
    try:
        g_state = th.load(ckpt_path, map_location="cpu")["G"]
        if "stage1.conv1.conv.weight" in g_state:
            from model.networks import Generator
        else:
            from model.networks_tf import Generator
        gen = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
        gen.load_state_dict(g_state, strict=True)
        gen.eval()
    finally:
        if dfv2_path in sys.path:
            sys.path.remove(dfv2_path)
    return gen


def inpaint_deepfillv2(generator, gt, keep_mask, device):
    """
    DeepFillV2 inpainting.
    gt        : [1,3,H,W] in [-1,1]
    keep_mask : [1,1,H,W]  1=keep, 0=inpaint  (RePaint convention)
    Returns   : [1,3,H,W] in [-1,1]
    """
    # DeepFill convention: mask=1 → masked (inpaint), mask=0 → visible (keep)
    dfill_mask   = (1.0 - keep_mask).to(device)
    image        = gt.to(device)
    image_masked = image * (1.0 - dfill_mask)
    ones         = th.ones_like(image_masked)[:, :1]
    x_in         = th.cat([image_masked, ones, ones * dfill_mask], dim=1)

    with th.inference_mode():
        _, x_stage2 = generator(x_in, dfill_mask)

    return image * (1.0 - dfill_mask) + x_stage2 * dfill_mask



def main():
    conf = conf_mgt.conf_base.Default_Conf()
    conf.update(yamlread(CONF_PATH))

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── RePaint model
    print("Loading RePaint diffusion model…")
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
    print("RePaint model ready.")

    # ── Guidance classifier
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
                x_in   = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                logp   = F.log_softmax(logits, dim=-1)
                sel    = logp[range(len(logits)), y.view(-1)]
                return th.autograd.grad(sel.sum(), x_in)[0] * conf.classifier_scale

        print("Guidance classifier ready.")

    def model_fn(x, t, y=None, gt=None, **_):
        return model(x, t, y if conf.class_cond else None, gt=gt)

    # ── DeepFillV2 
    print("Loading DeepFillV2…")
    deepfill = load_deepfillv2(DEEPFILL_CKPT, device)
    print("DeepFillV2 ready.")

    # ── VGG16 for XAI 
    print("Loading VGG16 for XAI attribution…")
    vgg = load_vgg16(VGG_PATH)
    print("VGG16 ready.")

    # ── Load first image 
    eval_name = conf.get_default_eval_name()
    dl        = conf.get_dataloader(dset="eval", dsName=eval_name)
    batch     = next(iter(dl))
    for key, val in batch.items():
        if isinstance(val, th.Tensor):
            batch[key] = val.to(device)

    gt       = batch["GT"]                        # [1,3,H,W] in [-1,1]
    img_name = batch["GT_name"][0].split(".")[0]
    img_np   = to_uint8(gt)[0]                    # HWC uint8
    print(f"Image: {img_name}  shape: {img_np.shape}")

    # ── XAI attribution & masks 
    method_xai = conf.get("method_xai", "saliency")
    print(f"XAI method: {method_xai}  |  mask: {int(MASK_PCT*100)}%")

    aurc_keep, auic_keep = build_masks(img_np, method_xai, MASK_PCT, vgg)
    aurc_keep_dev = aurc_keep.to(device)
    auic_keep_dev = auic_keep.to(device)

    # ── Masked images 
    masked_aurc_np = to_uint8(make_masked_image(gt, aurc_keep_dev))[0]
    masked_auic_np = to_uint8(make_masked_image(gt, auic_keep_dev))[0]

    # ── RePaint 
    print("RePaint AURC…")
    repaint_aurc_np = to_uint8(
        inpaint_repaint(gt, aurc_keep_dev, model_fn, diffusion, conf, cond_fn, device)
    )[0]

    print("RePaint AUIC…")
    repaint_auic_np = to_uint8(
        inpaint_repaint(gt, auic_keep_dev, model_fn, diffusion, conf, cond_fn, device)
    )[0]

    # ── DeepFillV2 
    print("DeepFillV2 AURC…")
    dfv2_aurc_np = to_uint8(
        inpaint_deepfillv2(deepfill, gt, aurc_keep, device)
    )[0]

    print("DeepFillV2 AUIC…")
    dfv2_auic_np = to_uint8(
        inpaint_deepfillv2(deepfill, gt, auic_keep, device)
    )[0]

    # ── Save 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outputs = {
        "GT":              img_np,
        "masked_aurc":     masked_aurc_np,
        "masked_auic":     masked_auic_np,
        "repaint_aurc":    repaint_aurc_np,
        "repaint_auic":    repaint_auic_np,
        "deepfillv2_aurc": dfv2_aurc_np,
        "deepfillv2_auic": dfv2_auic_np,
    }
    for fname, arr in outputs.items():
        path = os.path.join(OUTPUT_DIR, f"{fname}.png")
        Image.fromarray(arr).save(path)
        print(f"  saved → {path}")

    print(f"\nDone. All results in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
