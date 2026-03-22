import os
import argparse
import json
import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

import conf_mgt
from utils import yamlread
import AE_inet
from pixel_wise_comp import count_identical_adjacent

CLASS_INDEX_PATH = "./data/weights/imagenet_class_index.json"
VGG_PATH         = "data/weights/vgg16-397923af.pth"
AE_PATH          = "./data/weights/inet_AE.pth"
INPUT_ROOT       = "./latest_outputs"
PCT_DELETED      = 0.30
VARIANTS         = ["masked", "blurred", "repainted", "deepfill"]



def load_vgg(path):
    model = models.vgg16(weights=None)
    model.load_state_dict(th.load(path, map_location="cpu"))
    model.eval()
    return model


class LogitEvaluator:
    def __init__(self, device, class_index_path):
        self.device = device
        with open(class_index_path, "r") as f:
            idx_map = json.load(f)
        self.idx2label = [idx_map[str(i)][1] for i in range(len(idx_map))]
        self.model = load_vgg(VGG_PATH).to(device)
        self.tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def evaluate(self, img_pil: Image.Image) -> np.ndarray:
        t = self.tf(img_pil).unsqueeze(0).to(self.device)
        with th.no_grad():
            logits = self.model(t)
        return logits.cpu().numpy().flatten()

    def predict(self, logits: np.ndarray):
        idx = int(np.argmax(logits))
        return idx, self.idx2label[idx]

    @staticmethod
    def softmax_prob(logits: np.ndarray, idx: int) -> float:
        probs = F.softmax(th.from_numpy(logits), dim=0)
        return float(probs[idx].item())


def compute_ae_loss(ae, img_pil: Image.Image, device) -> float:
    tf_ae = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    img_tensor = tf_ae(img_pil).unsqueeze(0).to(device)
    with th.no_grad():
        recon = ae(img_tensor)
        loss = F.mse_loss(recon, img_tensor, reduction="mean")
    return float(loss.item())


def l2_distance(img1_np: np.ndarray, img2_np: np.ndarray) -> float:
    return float(np.sqrt(np.sum((img1_np.astype(np.float32) - img2_np.astype(np.float32)) ** 2)))


def evaluate_variant(img_pil: Image.Image, gt_np: np.ndarray, gt_idx: int,
                     evaluator: LogitEvaluator, ae, device) -> dict:
    logits  = evaluator.evaluate(img_pil)
    prob    = evaluator.softmax_prob(logits, gt_idx)
    logit   = float(logits[gt_idx])
    l2      = l2_distance(np.array(img_pil), gt_np)
    ae_loss = compute_ae_loss(ae, img_pil, device)
    identical = count_identical_adjacent(img_pil)
    return {
        "prob":                prob,
        "logit":               logit,
        "l2_dist":             l2,
        "ae_loss":             ae_loss,
        "identical_adjacent":  list(identical),
    }


def main(conf: conf_mgt.Default_Conf):
    print("Start AURC Evaluator")

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print("Device:", device)

    evaluator = LogitEvaluator(device=device, class_index_path=CLASS_INDEX_PATH)
    ae = AE_inet.get_AE(AE_PATH, device=device)
    print("Models loaded\n")

    gt_folder  = os.path.join(INPUT_ROOT, "GT")
    output_dir = os.path.join(INPUT_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)

    gt_files = sorted([
        f for f in os.listdir(gt_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"Found {len(gt_files)} GT images\n")

    for gt_fname in gt_files:
        img_name = os.path.splitext(gt_fname)[0]
        print(f"Evaluating: {img_name}")

        gt_pil = Image.open(os.path.join(gt_folder, gt_fname)).convert("RGB")
        gt_np  = np.array(gt_pil)

        # GT reference metrics
        gt_logits              = evaluator.evaluate(gt_pil)
        gt_idx, gt_label       = evaluator.predict(gt_logits)
        gt_prob                = evaluator.softmax_prob(gt_logits, gt_idx)
        gt_ae_loss             = compute_ae_loss(ae, gt_pil, device)
        gt_identical           = count_identical_adjacent(gt_pil)

        print(f"  GT class: {gt_label} (idx={gt_idx}), prob={gt_prob:.4f}, ae_loss={gt_ae_loss:.4f}")

        record = {
            "img_name":              img_name,
            "pct_deleted":           PCT_DELETED,
            "gt_class_idx":          gt_idx,
            "gt_class_label":        gt_label,
            "gt_prob":               gt_prob,
            "gt_logit":              float(gt_logits[gt_idx]),
            "gt_ae_loss":            gt_ae_loss,
            "gt_identical_adjacent": list(gt_identical),
        }

        # Evaluate each variant
        for variant in VARIANTS:
            variant_path = os.path.join(INPUT_ROOT, variant, gt_fname)
            if not os.path.exists(variant_path):
                print(f"  [{variant}] not found — skipping")
                record[variant] = None
                continue

            variant_pil = Image.open(variant_path).convert("RGB")
            metrics     = evaluate_variant(variant_pil, gt_np, gt_idx, evaluator, ae, device)
            record[variant] = metrics

            print(
                f"  [{variant}] prob={metrics['prob']:.4f}, logit={metrics['logit']:.4f}, "
                f"l2={metrics['l2_dist']:.2f}, ae_loss={metrics['ae_loss']:.4f}"
            )

        json_path = os.path.join(output_dir, f"{img_name}_eval.json")
        with open(json_path, "w") as f:
            json.dump(
                record, f, indent=4, sort_keys=True,
                default=lambda o: o.tolist() if hasattr(o, "tolist")
                                  else float(o) if hasattr(o, "item")
                                  else o,
            )
        print(f"  Saved {json_path}\n")

    print(f"Evaluator done. Results in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    conf = conf_mgt.conf_base.Default_Conf()
    conf.update(yamlread(args.conf_path))
    main(conf)
