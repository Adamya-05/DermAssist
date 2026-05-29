# preprocess_dermo.py
import os, cv2, numpy as np, pandas as pd, torch
import segmentation_models_pytorch as smp
from skimage.morphology import remove_small_objects
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from sklearn.cluster import KMeans
import math

print("\n=== DERMOSCOPIC PREPROCESSING STARTED ===\n")

# -------------------------------------------------------------
# 1) PATHS (EDIT THESE FOR YOUR LOCAL MACHINE)
# -------------------------------------------------------------
ROOT = os.getcwd()

IMAGE_DIR = os.path.join(ROOT, "images")  # your dermo+clinical images folder
TRIP_CSV = os.path.join(ROOT, "triplet_with_labels.csv")

OUT_PRE = os.path.join(ROOT, "preprocessed_dermo")
OUT_MASK = os.path.join(ROOT, "masks_dermo")
OUT_ROI = os.path.join(ROOT, "rois_dermo")
OUT_DIP = os.path.join(ROOT, "dip_features_dermo.csv")

os.makedirs(OUT_PRE, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)
os.makedirs(OUT_ROI, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# -------------------------------------------------------------
# 2) LOAD DERMO LIST
# -------------------------------------------------------------
trip = pd.read_csv(TRIP_CSV)

dermo_list = trip["dermo"].dropna().unique().tolist()
print("Found dermoscopic images:", len(dermo_list))

# -------------------------------------------------------------
# 3) DIP FUNCTIONS (YOUR EXACT CODE)
# -------------------------------------------------------------

def grabcut_refine(img_bgr, init_mask_binary, iter_count=5):
    h, w = init_mask_binary.shape
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    dist = cv2.distanceTransform((init_mask_binary>0).astype('uint8'), cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.6 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype('uint8')
    gc_mask[sure_fg==255] = cv2.GC_FGD

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    sure_bg = cv2.dilate((init_mask_binary==0).astype('uint8')*255, kernel, iterations=3)
    gc_mask[sure_bg==255] = cv2.GC_BGD

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(img_bgr, gc_mask, None, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_MASK)

    mask2 = np.where((gc_mask==cv2.GC_FGD) | (gc_mask==cv2.GC_PR_FGD), 1, 0).astype('uint8')
    result_mask = (mask2 * 255).astype('uint8')

    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return result_mask

def otsu_segmentation(img_bgr, min_area=200):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    th_bool = th.astype(bool)
    th_bool = remove_small_objects(th_bool, min_size=min_area)
    return (th_bool * 255).astype('uint8')

def asymmetry_score(mask):
    mask = (mask>0).astype(np.uint8)
    M = cv2.moments(mask)
    if M['m00'] == 0:
        return 1.0
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # vertical split
    left = mask[:, :cx]
    right = mask[:, cx:]
    diff = abs(left.shape[1] - right.shape[1])
    if diff>0:
        if left.shape[1] < right.shape[1]:
            left = np.pad(left, ((0,0),(diff,0)))
        else:
            right = np.pad(right, ((0,0),(diff,0)))
    overlap_v = np.sum(np.fliplr(left) & right)
    area = np.sum(mask)+1e-9
    asym_v = 1 - (overlap_v / area)

    # horizontal split
    top = mask[:cy, :]
    bottom = mask[cy:, :]
    diff2 = abs(top.shape[0] - bottom.shape[0])
    if diff2>0:
        if top.shape[0] < bottom.shape[0]:
            top = np.pad(top, ((diff2,0),(0,0)))
        else:
            bottom = np.pad(bottom, ((diff2,0),(0,0)))
    overlap_h = np.sum(np.flipud(top) & bottom)
    asym_h = 1 - (overlap_h / area)

    return float((asym_v + asym_h)/2)

def border_metrics(img_bgr, mask):
    mask_u = (mask>0).astype('uint8')
    cnts, _ = cv2.findContours(mask_u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt) + 1e-9
    compactness = (peri**2) / (4 * math.pi * area)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    ring = cv2.dilate(mask_u, kernel) - cv2.erode(mask_u, kernel)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1,0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0,1)
    grad = np.sqrt(gx**2 + gy**2)
    mg = float(np.mean(grad[ring==1])) if np.any(ring==1) else 0.0

    return float(compactness), mg

def color_clusters(img_bgr, mask, k=3):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab[mask>0]
    if len(pixels)==0:
        return None
    km = KMeans(n_clusters=k, random_state=0).fit(pixels)
    cts = np.bincount(km.labels_)
    return (cts/cts.sum()).tolist()

def glcm_features(img_bgr, mask):
    gray = rgb2gray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    gray_u8 = img_as_ubyte(gray)
    lesion = gray_u8.copy()
    lesion[mask==0] = 0

    cnts,_ = cv2.findContours((mask>0).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x,y,w,h = cv2.boundingRect(cnts[0])
    crop = lesion[y:y+h, x:x+w]
    if crop.size==0:
        return None

    levels = 64
    crop_q = (crop/4).astype(np.uint8)

    glcm = greycomatrix(crop_q, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
    return {
        'contrast': float(greycoprops(glcm,'contrast')[0,0]),
        'energy': float(greycoprops(glcm,'energy')[0,0]),
        'homogeneity': float(greycoprops(glcm,'homogeneity')[0,0])
    }


# -------------------------------------------------------------
# 4) PREPROCESSING HELPERS
# -------------------------------------------------------------

def hair_removal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(17,17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    _, mask = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 3)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

def apply_clahe(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); h,s,v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    return cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)

def preprocess_dermo(img):
    clean = hair_removal(img)
    clean = apply_clahe(clean)
    return cv2.resize(clean, (512,512))

# -------------------------------------------------------------
# 5) Load U-Net for segmentation
# -------------------------------------------------------------
unet = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
).to(device)
unet.eval()

def unet_mask(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (256,256)).astype('float32')/255.0
    inp = np.transpose(inp,(2,0,1))
    tens = torch.tensor(inp).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = unet(tens)
    pred = torch.sigmoid(pred)[0][0].cpu().numpy()
    pred = cv2.resize(pred, (img_bgr.shape[1], img_bgr.shape[0]))
    return (pred>0.5).astype('uint8')*255

# -------------------------------------------------------------
# 6) PROCESS ALL DERMO IMAGES
# -------------------------------------------------------------
rows=[]
missing=0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

for fname in dermo_list:
    src = os.path.join(IMAGE_DIR, fname)
    if not os.path.exists(src):
        missing+=1
        continue

    img = cv2.imread(src)
    pre = preprocess_dermo(img)
    mask = unet_mask(pre)

    if mask.sum()==0:
        init = otsu_segmentation(pre)
        mask = grabcut_refine(pre, init)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    roi = cv2.bitwise_and(pre, pre, mask=mask)

    # ---- SAVE FILES ----
    cv2.imwrite(os.path.join(OUT_PRE, fname), pre)
    cv2.imwrite(os.path.join(OUT_MASK, fname.replace('.jpg','_mask.png')), mask)
    cv2.imwrite(os.path.join(OUT_ROI, fname.replace('.jpg','_roi.png')), roi)

    # ---- DIP FEATURES ----
    asym = asymmetry_score(mask)
    compact, mg = border_metrics(pre, mask)
    colors = color_clusters(pre, mask)
    glcm = glcm_features(pre, mask)

    rows.append({
        'file': fname,
        'asymmetry': asym,
        'compactness': compact,
        'mean_grad': mg,
        'color_props': colors,
        'glcm': glcm
    })

print("Missing dermo images:", missing)
pd.DataFrame(rows).to_csv(OUT_DIP, index=False)
print("\nSaved DIP features to:", OUT_DIP)
print("\n=== DERMOSCOPIC PREPROCESSING COMPLETED ===\n")
