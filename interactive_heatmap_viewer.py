import streamlit as st
import torch
import os
import sys
import numpy as np
import cv2
import h5py
import pandas as pd
from PIL import Image
import openslide
from scipy.ndimage import gaussian_filter

# Add CHIEF to path
sys.path.insert(0, os.path.join(os.getcwd(), 'CHIEF'))
sys.path.insert(0, os.path.join(os.getcwd(), 'IPD-Brain'))

st.set_page_config(layout="wide", page_title="WSI Heatmap Explorer")

# =============================
# HELPER FUNCTIONS
# =============================

@st.cache_resource
def load_models(model_type='chief'):
    """Load and cache model components"""
    from CHIEF.Model.network import Classifier_1fc, DimReduction
    from CHIEF.Model.Attention import Attention_Gated
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model_type == 'chief':
        # CHIEF uses 768 input channels
        classifier = Classifier_1fc(384, 3).to(DEVICE)
        dimReduction = DimReduction(768, 384).to(DEVICE)
        attention = Attention_Gated(L=384, D=128, K=1).to(DEVICE)
    elif model_type == 'focal':
        # IPD-Brain focal uses 2048 input channels
        classifier = Classifier_1fc(384, 3).to(DEVICE)
        dimReduction = DimReduction(2048, 384).to(DEVICE)
        attention = Attention_Gated(L=384, D=128, K=1).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return classifier, dimReduction, attention, DEVICE


@st.cache_resource
def load_checkpoint(model_path, model_type='chief'):
    """Load model weights from checkpoint"""
    classifier, dimReduction, attention, device = load_models(model_type)
    
    if not os.path.exists(model_path):
        return None, None, None, None, None, f"Model not found: {model_path}"
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        def clean(sd):
            return {k.replace("module.", ""): v for k, v in sd.items()}
        
        classifier.load_state_dict(clean(checkpoint['classifier']))
        dimReduction.load_state_dict(clean(checkpoint['dim_reduction']))
        attention.load_state_dict(clean(checkpoint['attention']))
        
        classifier.eval()
        dimReduction.eval()
        attention.eval()
        
        return classifier, dimReduction, attention, device, model_path, None
    except Exception as e:
        return None, None, None, None, None, f"Error loading model: {str(e)}"


def get_feature_file_path(slide_id, data_dir):
    """Construct path to feature file"""
    return os.path.join(data_dir, f"{slide_id}.pt")


def get_h5_file_path(slide_id, h5_dir):
    """Construct path to h5 coordinates file"""
    return os.path.join(h5_dir, f"{slide_id}.h5")


def get_svs_file_path(slide_id, svs_dir):
    """Construct path to SVS slide file"""
    for file in os.listdir(svs_dir):
        if slide_id in file and file.endswith('.svs'):
            return os.path.join(svs_dir, file)
    return None

def get_true_label(slide_id, csv_path):
    """Read the actual label from the CSV mapping"""
    try:
        df = pd.read_csv(csv_path)
        # Search for true label using slide_id matching
        row = df[df['slide_id'].str.contains(slide_id, na=False)]
        if not row.empty:
            return row.iloc[0]['label']
    except Exception:
        pass
    return "Unknown"


def generate_heatmap(classifier, dimReduction, attention, device, 
                     feats, coords, svs_path, threshold_percentile):
    """Generate heatmap from attention weights"""
    
    # Load slide and get thumbnail
    slide_obj = openslide.OpenSlide(svs_path)
    w_wsi, h_wsi = slide_obj.dimensions
    
    THUMB_SIZE = 2500
    thumbnail = slide_obj.get_thumbnail((THUMB_SIZE, THUMB_SIZE))
    thumb_np = np.array(thumbnail.convert("RGB"))
    
    thumb_h, thumb_w = thumb_np.shape[:2]
    scale_x = thumb_w / w_wsi
    scale_y = thumb_h / h_wsi
    
    # Get attention weights
    with torch.no_grad():
        feats = feats.to(device)
        if len(feats.shape) == 4:
            feats = feats.mean(dim=[1, 2])
        elif len(feats.shape) > 2:
            feats = feats.squeeze()
            
        # Optional: apply normalization if the features weren't already normalized
        # feats = (feats - feats.mean(dim=0, keepdim=True)) / (feats.std(dim=0, keepdim=True) + 1e-8)
        
        feats_reduced = dimReduction(feats)
        attn_logits = attention(feats_reduced).squeeze() # N
        
        # Compute prediction using CHIEF MIL aggregation logic
        tAA = torch.softmax(attn_logits, dim=0) # N
        tattFeats = torch.einsum('ns,n->ns', feats_reduced, tAA) # fs -> ns
        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0) # 1 x fs
        
        tPredict = classifier(tattFeat_tensor) # 1 x 3 classes
        pred_idx = torch.argmax(tPredict, dim=1).item()
        
        # Map label index to subtype
        target_map = {0: 'subtype_1', 1: 'subtype_2', 2: 'subtype_3'}
        pred_label = target_map.get(pred_idx, f"Class {pred_idx}")
        
        attn_vec = attn_logits.cpu().numpy().reshape(-1)
    
    # Create heatmap
    patch_size = 256
    heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    sx = max(1, int(patch_size * scale_x))
    sy = max(1, int(patch_size * scale_y))
    
    for i, (x, y) in enumerate(coords):
        if i >= len(attn_vec):
            break
        tx, ty = int(x * scale_x), int(y * scale_y)
        if ty + sy <= thumb_h and tx + sx <= thumb_w:
            heatmap[ty:ty+sy, tx:tx+sx] += attn_vec[i]
    
    # Smooth heatmap
    blurred = gaussian_filter(heatmap, sigma=sx * 3)
    
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    
    # Threshold
    valid_heat = blurred[blurred > 0]
    if len(valid_heat) == 0:
        return None, None, None, pred_label
    
    thresh_val = np.percentile(valid_heat, threshold_percentile)
    mask = blurred > thresh_val
    
    return blurred, mask, thumb_np, pred_label


def apply_visualization(thumb_np, blurred, mask, viz_mode='boundaries'):
    """Apply visualization to thumbnail"""
    overlay = thumb_np.copy()
    
    # Use dark blue (0, 0, 255 in RGB) instead of red for better visibility
    boundary_color = (0, 0, 255)  # Dark blue/purple
    
    if viz_mode == 'boundaries':
        # Draw dark blue boundaries only
        mask_binary = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, boundary_color, 4)
    else:
        # Heatmap overlay with consistent colormap (Viridis)
        # Use Viridis instead of Jet for better consistency and visibility
        cm = cv2.applyColorMap(np.uint8(blurred * 255), cv2.COLORMAP_VIRIDIS)
        cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
        overlay[mask] = (overlay[mask] * 0.3 + cm[mask] * 0.7).astype(np.uint8)
        
        # Draw dark blue boundaries
        mask_binary = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, boundary_color, 4)
    
    return overlay


# =============================
# STREAMLIT UI
# =============================

st.title("WSI Heatmap Explorer")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Configuration")
    
    model_type_display = st.radio("Model Type:", ["CHIEF", "IPD-Brain Focal"], horizontal=False)
    model_type = 'chief' if model_type_display == "CHIEF" else 'focal'
    
    model_path = st.text_input(
        "Model Path:",
        value="CHIEF/results/glioma_chief_run/glioma_chief_run_0/best_model.pth" if model_type == "chief" 
              else "/home/pathousr3/tcga_glioma_subset/IPD-Brain/results/ipd_300/ipd_300_0/best_model.pth"
    )
    
    data_dir = st.text_input(
        "Features Directory:",
        value="processed_chief/features/pt_files" if model_type == "chief"
              else "processed_pilot/features/pt_files"
    )
    
    h5_dir = st.text_input(
        "Coordinates Directory:",
        value="processed_pilot/patches"
    )
    
    svs_dir = st.text_input(
        "Slides Directory:",
        value="/mnt/nas/glioma_slides"
    )
    
    label_csv = st.text_input(
        "Labels CSV:",
        value="slides_labels_final.csv"
    )
    
    st.subheader("Upload Configuration")
    
    slide_id = st.text_input(
        "Slide ID:",
        placeholder="e.g., TCGA-02-0026-01Z-00-DX1.d8f3085f-e418-47da-86bc-20db44ac6efd"
    )

with col2:
    st.subheader("Slide Visualization")
    
    if not slide_id:
        st.info("Enter a slide ID in the configuration panel to begin")
    else:
        # Load model
        with st.spinner("Loading model..."):
            classifier, dimReduction, attention, device, loaded_path, error = load_checkpoint(model_path, model_type)
        
        if error:
            st.error(f"Error: {error}")
        else:
            st.success(f"Model loaded")
            
            # Check files exist
            feat_path = get_feature_file_path(slide_id, data_dir)
            h5_path = get_h5_file_path(slide_id, h5_dir)
            svs_path = get_svs_file_path(slide_id, svs_dir)
            
            if not os.path.exists(feat_path):
                st.error(f"Feature file not found: {feat_path}")
            elif not os.path.exists(h5_path):
                st.error(f"Coordinates file not found: {h5_path}")
            elif not svs_path:
                st.error(f"SVS file not found for slide: {slide_id}")
            else:
                # Load data
                with st.spinner("Loading slide data..."):
                    feats = torch.load(feat_path)
                    with h5py.File(h5_path, 'r') as f:
                        coords = f['coords'][:]
                
                # Visualization controls
                col_thresh, col_viz = st.columns(2)
                
                with col_thresh:
                    threshold = st.slider(
                        "Threshold Percentile:",
                        min_value=0,
                        max_value=100,
                        value=60,
                        step=1
                    )
                
                with col_viz:
                    viz_mode = st.radio(
                        "Visualization Mode:",
                        ["boundaries", "heatmap"],
                        horizontal=True
                    )
                
                # Generate and display
                with st.spinner("Generating heatmap..."):
                    blurred, mask, thumb_np, pred_label = generate_heatmap(
                        classifier, dimReduction, attention, device,
                        feats, coords, svs_path, threshold
                    )
                
                if blurred is None:
                    st.error("No valid attention weights found")
                else:
                    true_label = get_true_label(slide_id, label_csv)
                    
                    st.markdown("### Prediction Results")
                    col_pred1, col_pred2 = st.columns(2)
                    with col_pred1:
                        st.info(f"**Predicted Label:** {pred_label}")
                    with col_pred2:
                        st.info(f"**Actual Label:** {true_label}")
                    
                    if pred_label == true_label and true_label != "Unknown":
                        st.success("✅ Prediction Matches Actual Label!")
                    elif true_label != "Unknown":
                        st.warning("⚠️ Prediction Differs from Actual Label!")
                        
                    overlay = apply_visualization(thumb_np, blurred, mask, viz_mode)
                    
                    st.image(overlay, caption=f"Slide: {slide_id}", width=800)
                    
                    # Download button
                    img_pil = Image.fromarray(overlay)
                    from io import BytesIO
                    buf = BytesIO()
                    img_pil.save(buf, format="PNG")
                    buf.seek(0)
                    
                    st.download_button(
                        label="Download Image",
                        data=buf.getvalue(),
                        file_name=f"{slide_id}_heatmap.png",
                        mime="image/png"
                    )
