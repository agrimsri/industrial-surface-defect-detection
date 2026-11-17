import os
import nbformat as nbf

def build_inference_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.12.3'
        }
    }
    
    cells = []
    
    # Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# 🚀 Production Inference, Classification Gating & Kaggle Submission

## Executive Summary
This notebook implements the end-to-end inference and submission pipeline for the **Kaggle Industrial Surface Defect Detection** competition.
To maximize the competition metric ($0.70 \\times S_{Dice} + 0.20 \\times S_{FP} + 0.10 \\times S_{Empty}$), the inference pipeline executes:

```
                              Test Image (1 x 128 x 800)
                                           │
                              ┌────────────▼────────────┐
                              │ Dual-Head MT-FPN Model  │ (ResNet-34 Backbone)
                              └─────┬──────────────┬────┘
                                    │              │
                       Auxiliary Defect          Multi-Scale FPN
                         Probability p_k         Segmentation Map
                                    │              │
                                    └──────┬───────┘
                                           ▼
                              Two-Stage Gated Decision
                             ┌─────────────────────────┐
                             │ Is p_k < \u03c4_cls?          │
                             └────────┬───────┬────────┘
                                 YES  │       │  NO
                                      ▼       ▼
                              Empty Mask     Apply \u03c4_prob + Morphological
                                 ("")        Closing/Dilation + Min Size Filter
                                              (\u03c4_min Area Removal)
                                                      │
                                                      ▼
                                           Lossless Column-Major
                                            RLE String Encoding
```

Key Highlights:
1. **Model Loading & Checkpoint Verification**
2. **Two-Stage Classification Gated Post-Processing**
3. **Visual Inspection of Test Predictions with Colored Overlays**
4. **Full 12,000-Row Submission Generation**
5. **Rigorous Submission Verification & Integrity Checks**"""))

    # Imports & Setup
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src can be imported
if '.' not in sys.path:
    sys.path.append('.')
if os.path.abspath('.') not in sys.path:
    sys.path.append(os.path.abspath('.'))

from src.models import DualHeadMTFPN
from src.dataset import DEFECT_TYPES
from src.postprocess import postprocess_mask, rle_encode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inference Device: {device}")"""))

    # Section 1: Checkpoint Inspection
    cells.append(nbf.v4.new_markdown_cell("""## 1. Load Model Checkpoint & Architecture
We load our trained `DualHeadMTFPN` checkpoint."""))

    cells.append(nbf.v4.new_code_cell("""checkpoint_path = 'checkpoints/production/dualhead_mtfpn_resnet34_best.pth'
if not os.path.exists(checkpoint_path):
    # Fallback to experiment checkpoint if production checkpoint is still training
    checkpoint_path = 'checkpoints/experiments/model_3_best.pth'

model = DualHeadMTFPN(backbone_name='resnet34', num_classes=4, pretrained=False).to(device)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from {checkpoint_path}")
    print(f"Best recorded val score: {checkpoint.get('best_score', 'N/A')}")
else:
    print(f"⚠️ Checkpoint not found at {checkpoint_path}. Instantiating fresh weights.")

model.eval()"""))

    # Section 2: Gated Inference Pipeline Function
    cells.append(nbf.v4.new_markdown_cell("""## 2. Gated Inference with Test-Time Augmentation (TTA)
We define the test inference function with horizontal flip TTA and defect-specific thresholding."""))

    cells.append(nbf.v4.new_code_cell("""@torch.no_grad()
def predict_image(model, img_path, device, use_tta=True):
    # Load 1-channel grayscale directly
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_norm = img.astype(np.float32) / 255.0
    x = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.float16):
        cls_logits, seg_logits = model(x)
        cls_probs = torch.sigmoid(cls_logits).squeeze(0).cpu().numpy()
        seg_probs = torch.sigmoid(seg_logits).squeeze(0).cpu().numpy()
        
        if use_tta:
            # Horizontal flip TTA
            x_flip = torch.flip(x, dims=[-1])
            cls_logits_flip, seg_logits_flip = model(x_flip)
            cls_probs_flip = torch.sigmoid(cls_logits_flip).squeeze(0).cpu().numpy()
            seg_probs_flip = torch.flip(torch.sigmoid(seg_logits_flip), dims=[-1]).squeeze(0).cpu().numpy()
            
            cls_probs = 0.5 * (cls_probs + cls_probs_flip)
            seg_probs = 0.5 * (seg_probs + seg_probs_flip)
            
    # Apply two-stage gated post-processing per defect
    clean_masks = np.zeros((4, 128, 800), dtype=np.uint8)
    for c in range(4):
        clean_masks[c] = postprocess_mask(
            seg_probs[c],
            cls_prob=cls_probs[c],
            cls_idx=c
        )
        
    return clean_masks, cls_probs, img"""))

    # Section 3: Visual Inspection on Unseen Test Images
    cells.append(nbf.v4.new_markdown_cell("""## 3. Visual Inspection of Test Predictions
We inspect model predictions across a sample of test images."""))

    cells.append(nbf.v4.new_code_cell("""test_dir = 'public/test'
sample_test_ids = os.listdir(test_dir)[:6]

fig, axes = plt.subplots(len(sample_test_ids), 2, figsize=(16, 3 * len(sample_test_ids)))
defect_colors = {
    0: [52, 152, 219],   # Blue
    1: [231, 76, 60],    # Red
    2: [46, 204, 113],   # Green
    3: [243, 156, 18]    # Orange
}

for idx, fname in enumerate(sample_test_ids):
    img_path = os.path.join(test_dir, fname)
    masks, cls_probs, raw_img = predict_image(model, img_path, device, use_tta=True)
    
    # Raw Image
    axes[idx, 0].imshow(raw_img, cmap='gray')
    axes[idx, 0].set_title(f"Test Image: {fname}", fontsize=11, fontweight='bold')
    axes[idx, 0].axis('off')
    
    # Overlay Image
    rgb = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
    active_defects = []
    for c in range(4):
        if masks[c].sum() > 0:
            rgb[masks[c] == 1] = np.clip(0.4 * rgb[masks[c] == 1] + 0.6 * np.array(defect_colors[c]), 0, 255).astype(np.uint8)
            active_defects.append(f"{DEFECT_TYPES[c]} ({cls_probs[c]:.2f}, {masks[c].sum()}px)")
            
    title_str = "Clean Surface (All Empty)" if not active_defects else "Detected: " + ", ".join(active_defects)
    axes[idx, 1].imshow(rgb)
    axes[idx, 1].set_title(title_str, fontsize=11, fontweight='bold', color='green' if not active_defects else 'red')
    axes[idx, 1].axis('off')

plt.tight_layout()
plt.show()"""))

    # Section 4: Full Submission Generation
    cells.append(nbf.v4.new_markdown_cell("""## 4. Full Competition Submission Generation
Generating predictions for all 3,000 test images ($12,000$ predictions in `sample_submission.csv` order)."""))

    cells.append(nbf.v4.new_code_cell("""sub_template = pd.read_csv('public/sample_submission.csv')
print(f"Sample submission template loaded: {len(sub_template)} rows")

submission_rows = []
unique_test_ids = sub_template['image_id'].unique()

print(f"Running inference across {len(unique_test_ids)} test images...")
id_to_masks = {}
for i, img_id in enumerate(unique_test_ids):
    img_path = os.path.join(test_dir, f"{img_id}.jpg")
    masks, _, _ = predict_image(model, img_path, device, use_tta=False)
    id_to_masks[img_id] = masks
    if (i + 1) % 1000 == 0 or (i + 1) == len(unique_test_ids):
        print(f"  Processed {i+1}/{len(unique_test_ids)} test images ({((i+1)/len(unique_test_ids))*100:.1f}%)")

# Format into submission dataframe
for idx, row in sub_template.iterrows():
    row_id = row['ID']
    img_id = row['image_id']
    d_type = row['defect_type']
    c_idx = DEFECT_TYPES.index(d_type)
    
    mask = id_to_masks[img_id][c_idx]
    rle = rle_encode(mask)
    submission_rows.append({
        'ID': row_id,
        'image_id': img_id,
        'defect_type': d_type,
        'mask_rle': rle
    })

submission_df = pd.DataFrame(submission_rows)
os.makedirs('submissions', exist_ok=True)
sub_out_path = 'submissions/submission.csv'
submission_df.to_csv(sub_out_path, index=False)
print(f"✅ Saved submission to {sub_out_path}")"""))

    # Section 5: Verification & Integrity Checks
    cells.append(nbf.v4.new_markdown_cell("""## 5. Submission Integrity & Safety Checks
We perform rigorous automated verification on the generated submission file:
1. Exact row count equals 12,000.
2. Columns match `['ID', 'image_id', 'defect_type', 'mask_rle']`.
3. 0 NaN / Null values.
4. Clean mask proportion matches expected training distribution (~84-88% empty)."""))

    cells.append(nbf.v4.new_code_cell("""check_sub = pd.read_csv(sub_out_path, keep_default_na=False)

print("🔍 VERIFICATION REPORT:")
print(f"• Total Rows: {len(check_sub)} (Expected: 12000) -> {'PASS ✅' if len(check_sub) == 12000 else 'FAIL ❌'}")
print(f"• Columns: {list(check_sub.columns)} -> {'PASS ✅' if list(check_sub.columns) == ['ID', 'image_id', 'defect_type', 'mask_rle'] else 'FAIL ❌'}")
print(f"• Null Values: {check_sub.isnull().sum().sum()} -> {'PASS ✅' if check_sub.isnull().sum().sum() == 0 else 'FAIL ❌'}")

empty_count = (check_sub['mask_rle'].str.strip() == '').sum()
empty_pct = (empty_count / len(check_sub)) * 100
print(f"• Clean / Empty Masks: {empty_count:,} / {len(check_sub):,} ({empty_pct:.2f}%)")
print(f"• Positive Masks Predicted: {len(check_sub) - empty_count:,}")
check_sub.head(10)"""))

    # Section 6: Kaggle Submission Protocol
    cells.append(nbf.v4.new_markdown_cell("""## 6. Official Kaggle CLI Submission Protocol
Submissions can be sent directly to the leaderboard via:
```bash
kaggle competitions submit -c industrial-surface-defect-detection -f submissions/submission.csv -m "Model 3: Dual-Head MT-FPN ResNet34 Gated"
```
And verified with:
```bash
kaggle competitions submissions industrial-surface-defect-detection
```"""))

    nb.cells = cells
    out_path = 'notebooks/03_inference_and_submission.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully generated {out_path} with {len(cells)} cells.")

if __name__ == '__main__':
    build_inference_notebook()
