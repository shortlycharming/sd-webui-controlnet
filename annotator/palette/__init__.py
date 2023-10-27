import os
import numpy as np
from PIL import Image
from modules import devices
from annotator.annotator_path import models_path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def normalize(data: Image.Image):
    result = np.array(data.convert("RGB"), dtype=np.float32)
    result = (result / 255.0 - 0.5) / 0.5 * 255.0
    return result

def get_cond_color(cond_image, mask_size=8):
    cond_image = Image.fromarray(cond_image)
    H, W = cond_image.size
    cond_image = cond_image.resize((W // mask_size, H // mask_size), Image.BICUBIC)
    color = cond_image.resize((H, W), Image.NEAREST)
    return color

def apply_rectangular_palette(*args, **kwargs):
    result = get_cond_color(*args, **kwargs)
    result = normalize(result)
    return result

def show_anns(anns, cond_image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)

    h, w = sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]

    palette = np.zeros((h, w, 3))
    # mask = np.ones((h, w, 3)).astype(np.float64)
    visited = np.zeros((h, w))
    for ann in sorted_anns:
        m = ann['segmentation']
        modify_m = (m * (1 - visited)) == 1
        if modify_m.sum() > 0:
            this_color = np.mean(cond_image[modify_m], 0)
            palette[modify_m] += this_color
            visited[modify_m] += 1
        ann.pop('segmentation')

    palette = Image.fromarray(palette.astype(np.uint8))
    return palette

class SAMImageAnnotator:
    model_dir = os.path.join(models_path, "palette")

    def __init__(self):
        self.model = None
        self.device = devices.get_device_for("controlnet")
        
    def load_model(self):
        remote_model_path = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        model_path = os.path.join(self.model_dir, "sam_vit_h_4b8939.pth")
        if not os.path.exists(model_path):
            from basicsr.utils.download_util import load_file_from_url
            load_file_from_url(remote_model_path, model_dir=self.model_dir)
        sam = sam_model_registry["default"](checkpoint=model_path)
        sam.to(device=devices.get_device_for("controlnet"))
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.1,
            stability_score_thresh=0.1,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            points_per_batch=64,
        )

    def unload_model(self):
        if self.mask_generator is not None:
            self.mask_generator.predictor.model.cpu()

    def __call__(self, cond_image):
        if self.model is None:
            self.load_model()
        # cond_image = np.asarray(pil_image.convert("RGB"))
        masks = self.mask_generator.generate(cond_image)
        palette = show_anns(masks, cond_image)
        palette = palette.convert("RGB")
        return normalize(palette)
