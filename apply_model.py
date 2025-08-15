from main import gradient_descent, forwardprop, get_predictions
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt


def get_digit_slices(original_img):
    img = original_img.filter(ImageFilter.GaussianBlur(radius=0.6))
    img_array = np.array(img)
    dynamic_threshold = np.mean(img_array)
    binary_img = img.point(lambda p: 0 if p < dynamic_threshold - 25 else 255)
    binary_array = np.array(binary_img)

    column_sums = np.sum(binary_array, axis=0)

    slices = []
    in_digit = False
    start_col = 0

    for i in range(len(column_sums)):
        if column_sums[i] < 255 * binary_array.shape[0] and not in_digit:
            start_col = i
            in_digit = True
        elif column_sums[i] == 255 * binary_array.shape[0] and in_digit:
            end_col = i
            slices.append((start_col, end_col))
            in_digit = False

    if in_digit:
        slices.append((start_col, len(column_sums)))

    return slices


def preprocess_digit(img):
    bbox = img.getbbox()
    if not bbox:
        return None

    cropped_img = img.crop(bbox)
    width, height = cropped_img.size
    size = max(width, height) + 10
    padded_img = Image.new("L", (size, size), 255)
    padded_img.paste(cropped_img, ((size - width) // 2, (size - height) // 2))

    processed_digit = np.array(padded_img.resize((28, 28)))
    processed_digit = (255 - processed_digit) / 255.0

    return processed_digit.reshape(784, 1)


def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = (
        img.filter(ImageFilter.SHARPEN)
        .filter(ImageFilter.GaussianBlur(radius=0.8))
        .filter(ImageFilter.SMOOTH)
    )
    arr = np.array(img, dtype=np.uint8)
    med = np.median(arr)
    mean = np.mean(arr)
    #Otsu Algorithm idk whats happening
    dyn = int(round(0.6 * med + 0.42 * mean )) - 20
    if dyn < 5 or dyn > 250:
        hist, _ = np.histogram(arr, bins=256, range=(0, 256))
        total = arr.size
        sum_total = (np.arange(256) * hist).sum()
        sumB = 0
        wB = 0
        max_var = 0
        thresh = 120
        for t in range(256):
            wB += hist[t]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += t * hist[t]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            var_between = wB * wF * (mB - mF) ** 2
            if var_between > max_var:
                max_var = var_between
            thresh = t
        dyn = max(5, min(250, thresh - 8))

    binp = (arr < dyn).astype(np.uint8) * 255
    binp = (
        Image.fromarray(binp)
        .filter(ImageFilter.MaxFilter(3))
        .filter(ImageFilter.MinFilter(3))
    )
    img_resized = binp.resize((28, 28), resample=Image.BOX)
    img_array = np.array(img_resized, dtype=np.float32)
    img_normalized = (img_array / 255.0).astype(np.float32)
    img_final = img_normalized.reshape(784, 1)
    return img_final


def _center_of_mass_shift(arr28):
    ys, xs = np.nonzero(arr28 > 0)
    if ys.size == 0:
        return arr28
    cy = ys.mean()
    cx = xs.mean()
    shift_x = int(round(14 - cx))
    shift_y = int(round(14 - cy))
    if shift_x != 0:
        arr28 = np.roll(arr28, shift_x, axis=1)
        if shift_x > 0:
            arr28[:, :shift_x] = 0
        else:
            arr28[:, shift_x:] = 0
    if shift_y != 0:
        arr28 = np.roll(arr28, shift_y, axis=0)
        if shift_y > 0:
            arr28[:shift_y, :] = 0
        else:
            arr28[shift_y:, :] = 0
    return arr28


def segment_and_predict(
    image_path,
    debug=False,
    blur_radius=0.8,
    dyn_offset=20,
    col_thresh_frac=0.008,
    min_width=6,
    min_area_frac=0.0009,
    pad=6,
    merge_gap=6,
):
    """
    Loads The image as a greyscale, applies filters 
    Does Dynamic threshold or Otsu
    Tries to find a box 
    If found, Applies further noise cleaning (filtering) on them
    if not, uses the whole image
    normalize and run them through the model
    Also prints and shows per-segment images if debug=True.
    """
    # 1) open and apply same preprocessing logic up to binarization
    pil = Image.open(image_path).convert("L")
    pil = (
        pil.filter(ImageFilter.SHARPEN)
        .filter(ImageFilter.GaussianBlur(radius=blur_radius))
        .filter(ImageFilter.SMOOTH)
    )
    arr = np.array(pil, dtype=np.uint8)

    med = np.median(arr)
    mean = np.mean(arr)
    dyn = int(round(0.6 * med + 0.42 * mean)) - dyn_offset

    # fallback to simple Otsu-like if dyn nonsense
    if dyn < 5 or dyn > 250:
        hist, _ = np.histogram(arr, bins=256, range=(0, 256))
        total = arr.size
        sum_total = (np.arange(256) * hist).sum()
        sumB = 0
        wB = 0
        max_var = 0
        thresh = 128
        for t in range(256):
            wB += hist[t]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += t * hist[t]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            var_between = wB * wF * (mB - mF) ** 2
            if var_between > max_var:
                max_var = var_between
                thresh = t
        dyn = max(5, min(250, thresh - 8))

    bin_bool = arr < dyn
    bin_img = Image.fromarray((bin_bool.astype(np.uint8) * 255))
    
    bin_img = bin_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3)) 
    bin_arr = np.array(bin_img, dtype=np.uint8) 

    H, W = bin_arr.shape
    img_area = H * W
    min_area_px = max(1, int(min_area_frac * img_area))

    # vertical projection
    col_counts = (bin_arr > 128).sum(axis=0)
    col_thresh = max(1, int(col_thresh_frac * H))

    # find runs of columns that are "active"
    runs = []
    in_run = False
    s = 0
    for x in range(W):
        if col_counts[x] >= col_thresh:
            if not in_run:
                in_run = True
                s = x
        else:
            if in_run:
                runs.append((s, x - 1))
                in_run = False
    if in_run:
        runs.append((s, W - 1))

    # merge small gaps
    merged = []
    if runs:
        cs, ce = runs[0]
        for s, e in runs[1:]:
            if s - ce <= merge_gap:
                ce = e
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))

    boxes = []
    for x0, x1 in merged:
        window = bin_arr[:, x0 : x1 + 1] > 128
        if window.sum() == 0:
            continue
        row_counts = window.sum(axis=1)
        # vertical runs in window
        y_runs = []
        inr = False
        ys = 0
        for y in range(H):
            if row_counts[y] > 0:
                if not inr:
                    inr = True
                    ys = y
            else:
                if inr:
                    y_runs.append((ys, y - 1))
                    inr = False
        if inr:
            y_runs.append((ys, H - 1))
        if not y_runs:
            continue
        best = max(y_runs, key=lambda r: row_counts[r[0] : r[1] + 1].sum())
        y0, y1 = best
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        area = (bin_arr[y0 : y1 + 1, x0 : x1 + 1] > 128).sum()
        # filters
        if area < min_area_px or w < min_width or h < 8:
            continue
        xa = max(0, x0 - pad)
        ya = max(0, y0 - pad)
        xb = min(W, x1 + pad)
        yb = min(H, y1 + pad)
        boxes.append((xa, ya, xb - xa, yb - ya))

    # fallback: whole image
    if not boxes:
        boxes = [(0, 0, W, H)]

    # filter giant boxes (likely background)
    boxes = [b for b in boxes if not (b[2] > 0.95 * W and b[3] > 0.95 * H)]
    if not boxes:
        boxes = [(0, 0, W, H)]

    # create 28x28 images per box and predict
    preds = []
    preds_values = []
    for i, (x, y, w, h) in enumerate(sorted(boxes, key=lambda b: b[0])):
        crop = bin_arr[y : y + h, x : x + w]
        crop_u = (crop > 128).astype(np.uint8) * 255
        pil_crop = Image.fromarray(crop_u, mode="L")

        # resize to fit 20x20
        target = 20
        if w > h:
            new_w = target
            new_h = max(1, int(round(h * (target / w))))
        else:
            new_h = target
            new_w = max(1, int(round(w * (target / h))))
        pil_small = pil_crop.resize((new_w, new_h), resample=Image.NEAREST)

        canvas = Image.new("L", (28, 28), color=0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        canvas.paste(pil_small, (paste_x, paste_y))
        arr28 = np.array(canvas, dtype=np.uint8)

        arr28 = _center_of_mass_shift(arr28)

        final = arr28.astype(np.float32) / 255.0
        col = final.reshape(784, 1).astype(np.float32)

        # model prediction 
        _, _, _, a2 = forwardprop(w1, b1, w2, b2, col)
        p = get_predictions(a2)
        preds.append(str(p))
        preds_values.append(p)

        if debug:
            print(f"segment {i} box={(x, y, w, h)} -> {p}")
            plt.figure(figsize=(2, 2))
            plt.imshow(arr28, cmap="gray")
            plt.title(str(p))
            plt.axis("off")
            plt.show()

    combined = "".join(preds)
    return preds_values, combined


w1, b1, w2, b2 = gradient_descent(None, None, 0, 0, force_train=False)

if w1 is None or b1 is None or w2 is None or b2 is None:
    raise Exception(
        "Weights not initialized properly. Check if there are .npy files!\nIf there aren't, re-run main!"
    )

my_image_path = "./test.jpg"

pred_list, combined_str = segment_and_predict(my_image_path, debug=True)
print("Predictions list:", pred_list)
print("Combined reading:", combined_str)
