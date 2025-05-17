import torch
import open_clip

# from torchvision import transforms
from PIL import Image
from pathlib import Path
from .constants import CACHE_ROOT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# clip model + preprocessing
MODEL, PREP, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",  # loads original weights
    device=DEVICE,
)
MODEL.eval()

# text encoder
tokenizer = open_clip.get_tokenizer("ViT-B-32")
PROMPTS = [
    "runway photo: model wearing a floor-length evening gown",
    "runway photo: model wearing a knee-length A-line dress",
    "runway photo: model wearing a mini skirt",
    "runway photo: model wearing wide-leg trousers",
    "runway photo: model wearing a tailored blazer or coat",
    "runway photo: model wearing shorts",
]
with torch.no_grad():
    text_tokens = tokenizer(PROMPTS).to(DEVICE)
    text_features = MODEL.encode_text(text_tokens).float()


def clip_label(img_path: str):
    out_file = CACHE_ROOT / "clip" / (Path(img_path).stem + ".pt")
    if out_file.exists():
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(img_path).convert("RGB")
    image_feature = MODEL.encode_image(PREP(img).unsqueeze(0).to(DEVICE)).float()
    sims = (image_feature @ text_features.T).softmax(dim=-1)
    label_idx = int(sims.argmax(-1))
    torch.save({"vec": image_feature.cpu(), "label_idx": label_idx}, out_file)
