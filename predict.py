import tensorflow as tf
import numpy as np
import os
import json
import yaml
import cv2
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description='Predict using trained U-Net model on satellite imagery')
    parser.add_argument('--config_path', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--model_path', type=str, help='Path to trained model file (overrides config)')
    parser.add_argument('--weights_path', type=str, help='Path to trained weights file (overrides config)')
    parser.add_argument('--image_path', type=str, help='Path to input satellite image (overrides config)')
    parser.add_argument('--label_path', type=str, help='Path to ground truth label (overrides config)')
    parser.add_argument('--output_dir', type=str, help='Output directory for results (overrides config)')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size for prediction')
    parser.add_argument('--satellite', type=str, choices=['sn2', 'wv2'], 
                       help='Override satellite type (sn2 for Sentinel-2, wv2 for WorldView-2)')
    
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs found: {len(gpus)}")
        except RuntimeError as e:
            print(f"Error while setting memory growth: {e}")
    else:
        print("No GPUs found.")
    
    print(f"List of physical devices detected by TensorFlow: {tf.config.list_physical_devices()}")


def combined_loss(y_true, y_pred):
    def focal_loss_fn(y_true, y_pred, alpha=0.25, gamma=2.0):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1.0 - y_pred, gamma)
        loss_value = tf.reduce_sum(weight * cross_entropy, axis=-1)
        return loss_value
    
    def lovasz_grad(gt_sorted):
        gts = tf.reduce_sum(gt_sorted)
        intersection = gts - tf.cumsum(gt_sorted)
        union = gts + tf.cumsum(1.0 - gt_sorted)
        jaccard = 1.0 - intersection / union
        return tf.concat([jaccard[:1], jaccard[1:] - jaccard[:-1]], axis=0)
    
    def lovasz_softmax_flat(probas, labels):
        num_classes = probas.shape[1]
        losses = []
        for c in range(num_classes):
            fg = tf.cast(tf.equal(labels, c), probas.dtype)
            errors = tf.abs(fg - probas[:, c])
            errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0])
            fg_sorted = tf.gather(fg, perm)
            grad = lovasz_grad(fg_sorted)
            losses.append(tf.tensordot(errors_sorted, tf.stop_gradient(grad), axes=1))
        return tf.reduce_mean(losses)

    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
    
    ce_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
    focal_loss = tf.reduce_mean(focal_loss_fn(y_true_one_hot, y_pred))
    
    probas_flat = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
    labels_flat = tf.reshape(tf.cast(y_true, tf.int32), [-1])
    lovasz_loss = lovasz_softmax_flat(probas_flat, labels_flat)
    
    total_loss = ce_loss + 0.5 * focal_loss + 0.5 * lovasz_loss
    return total_loss


def get_satellite_bands(satellite_type):
    if satellite_type.lower() == 'wv2':
        return [2, 3, 5, 7]  # WorldView-2: Blue, Green, Red, NIR1
    else:  # Default to Sentinel-2
        return [1, 2, 3, 4]  # Sentinel-2: Blue, Green, Red, NIR


def load_tiff_image(image_path, satellite_type='sn2'):
    try:
        bands = get_satellite_bands(satellite_type)
        with rasterio.open(image_path) as dataset:
            height = dataset.height
            width = dataset.width
            image = dataset.read(bands)
            image = np.transpose(image, (1, 2, 0)) 
            profile = dataset.profile
            transform = dataset.transform
        
        print(f"Loaded image from {image_path}")
        print(f"Image shape: {image.shape}")
        print(f"Using bands: {bands}")
        
        return image, height, width, profile, transform
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None, None, None


def load_label_image(label_path):
    try:
        with rasterio.open(label_path) as dataset:
            label = dataset.read(1)  
            profile = dataset.profile
            transform = dataset.transform
        
        print(f"Loaded label from {label_path}")
        print(f"Label shape: {label.shape}")
        
        return label, profile, transform
    except Exception as e:
        print(f"Error loading label {label_path}: {e}")
        return None, None, None


def map_labels(labels, mapping, default=0):
    mapped_labels = np.full_like(labels, default)
    for original, new_idx in mapping.items():
        mapped_labels[labels == original] = new_idx
    return mapped_labels


def predict_patches(model, image, patch_size=256):
    """
    Predict on large satellite image by processing in patches
    
    Args:
        model: Tensorflow model
        image: Satellite image, shape (height, width, channels)
        patch_size: Size of patches to process
        
    Returns:
        Predicted label map with same height and width as input image
    """
    height, width = image.shape[:2]
    print(f"Input image dimensions: {height}x{width}")
    
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        print(f"Padding image by {pad_h} rows and {pad_w} columns")
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        padded_image = image
        
    h_patches = padded_image.shape[0] // patch_size
    w_patches = padded_image.shape[1] // patch_size
    total_patches = h_patches * w_patches
    
    print(f"Processing {total_patches} patches ({h_patches}x{w_patches}) of size {patch_size}x{patch_size}")
    
    predicted = np.zeros((padded_image.shape[0], padded_image.shape[1]), dtype=np.uint8)
    
    print("Starting prediction...")
    patch_count = 0
    
    for y in range(h_patches):
        for x in range(w_patches):
            patch_count += 1
            if patch_count % 10 == 0 or patch_count == total_patches:
                print(f"Processing patch {patch_count}/{total_patches}")
                
            patch = padded_image[
                y * patch_size:(y + 1) * patch_size,
                x * patch_size:(x + 1) * patch_size,
                :
            ]
            
            prediction = model.predict(np.expand_dims(patch, axis=0), verbose=0)
            pred_labels = np.argmax(prediction[0], axis=-1)
            
            predicted[
                y * patch_size:(y + 1) * patch_size,
                x * patch_size:(x + 1) * patch_size
            ] = pred_labels
    
    predicted = predicted[:height, :width]
    
    print(f"Prediction complete. Output shape: {predicted.shape}")
    return predicted


def normalize_rgb_4channel(image, lower_percentile=2, upper_percentile=98):
    if image.shape[-1] < 3:
        raise ValueError("Image does not have at least 3 channels.")
    
    rgb_image = image[:, :, :3]
    
    image_display = np.zeros_like(rgb_image, dtype=np.float32)
    for c in range(3):
        channel = rgb_image[:, :, c]
        p_low = np.percentile(channel, lower_percentile)
        p_high = np.percentile(channel, upper_percentile)
        image_display[:, :, c] = np.clip(
            (channel - p_low) / (p_high - p_low + 1e-8), 0, 1)
    
    return image_display


def labels_to_rgb(label, class_hex_colors):
    rgb_colors = np.array([mcolors.to_rgb(color) for color in class_hex_colors])
    rgb_image = rgb_colors[label]
    return rgb_image


def plot_prediction(sat_image, true_label, predicted_label, class_hex_colors, class_names, output_path):
    cmap = ListedColormap(class_hex_colors)
    norm = BoundaryNorm(np.arange(len(class_names) + 1), cmap.N)
    
    plt.figure(figsize=(24, 8))
    
    plt.subplot(1, 3, 1)
    plt.imshow(normalize_rgb_4channel(sat_image))
    plt.title('Satellite Image', fontsize=14)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    if true_label is not None:
        plt.imshow(true_label, cmap=cmap, norm=norm)
        plt.title('True Label', fontsize=14)
    else:
        plt.text(0.5, 0.5, 'No Ground Truth Available', 
                 ha='center', va='center', fontsize=14)
        plt.title('True Label (Not Available)', fontsize=14)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_label, cmap=cmap, norm=norm)
    plt.title('Predicted Label', fontsize=14)
    plt.axis('off')
    
    legend_patches = [Patch(color=class_hex_colors[i], label=class_names[i])
                     for i in range(len(class_names))]
    plt.figlegend(handles=legend_patches, loc='lower center', ncol=6, fontsize=10)
    
    plt.suptitle('Satellite Image Segmentation', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def save_geotiff(image, filename, profile, output_dir, colormap=None):
    filepath = os.path.join(output_dir, filename)
    
    out_profile = profile.copy()
    out_profile.update({
        'driver': 'GTiff',
        'dtype': 'uint8',
        'count': 1 if image.ndim == 2 else image.shape[-1],
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    })
    
    with rasterio.open(filepath, 'w', **out_profile) as dst:
        if image.ndim == 2:
            dst.write(image.astype('uint8'), 1)
            if colormap:
                # Convert hex colors to RGBA tuples
                rasterio_colormap = {
                    i: tuple(int(color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4)) + (255,)
                    for i, color in enumerate(colormap)
                }
                dst.write_colormap(1, rasterio_colormap)
        else:
            for i in range(image.shape[-1]):
                dst.write(image[:, :, i].astype('uint8'), i+1)
    
    print(f"Saved GeoTIFF to: {filepath}")
    return filepath


def calculate_metrics(true_label, pred_label, class_names):
    cm = confusion_matrix(true_label.flatten(), pred_label.flatten(), labels=range(len(class_names)))
    
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    
    IoU = intersection / np.maximum(union.astype(np.float32), 1e-6)
    mean_IoU = np.mean(IoU)
    
    precision_per_class = np.zeros(len(class_names))
    recall_per_class = np.zeros(len(class_names))
    
    for i in range(len(class_names)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_per_class[i] = precision
        recall_per_class[i] = recall
    
    mean_precision = np.mean(precision_per_class)
    mean_recall = np.mean(recall_per_class)
    
    freq = ground_truth_set / ground_truth_set.sum()
    frequency_weighted_IoU = (freq * IoU).sum()
    
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision_per_class,
        'Recall': recall_per_class,
        'IoU': IoU
    })
    
    summary_rows = pd.DataFrame([
        {
            'Class': 'Mean',
            'Precision': mean_precision,
            'Recall': mean_recall,
            'IoU': mean_IoU
        },
        {
            'Class': 'Frequency Weighted IoU',
            'Precision': None,
            'Recall': None,
            'IoU': frequency_weighted_IoU
        }
    ])
    
    metrics_df = pd.concat([metrics_df, summary_rows], ignore_index=True)
    
    return metrics_df, cm


def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(12, 10))
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized) 
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar=True, annot_kws={'size': 10}, vmin=0.0, vmax=1.0)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to: {output_path}")
    plt.close()


def plot_metrics(metrics_df, output_path):
    class_metrics = metrics_df[~metrics_df['Class'].isin(['Mean', 'Frequency Weighted IoU'])]
    
    classes = class_metrics['Class']
    precision = class_metrics['Precision'].astype(float)
    recall = class_metrics['Recall'].astype(float)
    iou = class_metrics['IoU'].astype(float)
    
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    plt.bar(x, recall, width, label='Recall', color='#3498db')
    plt.bar(x + width, iou, width, label='IoU', color='#e74c3c')
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Classification Metrics by Class')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics plot to: {output_path}")
    plt.close()


def main():
    args = parse_args()
    
    config = load_config(args.config_path)
    
    satellite_type = args.satellite if args.satellite else config.get('satellite_type', 'sn2')
    model_path = args.model_path if args.model_path else os.path.join(config['paths']['output_dir'], 'model.h5')
    weights_path = args.weights_path if args.weights_path else os.path.join(config['paths']['output_dir'], 'weights.h5')
    image_path = args.image_path if args.image_path else config.get('prediction', {}).get('image_path')
    label_path = args.label_path if args.label_path else config.get('prediction', {}).get('label_path')
    output_dir = args.output_dir if args.output_dir else config['paths']['output_dir']
    patch_size = args.patch_size if args.patch_size else config.get('prediction', {}).get('patch_size', 256)
    
    prediction_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(prediction_dir, exist_ok=True)
    
    class_names = config.get('class_names', [
        "Coral", "Coral Fore Reef", "Reef Crest - Coralline Algal Ridge",
        "Algae", "Seagrass", "Sediment & Rubble",
        "Terrestrial Vegetated", "Mangroves", "Deep Water",
        "Terrestrial Other & No Data", "Clouds"
    ])
    
    class_hex_colors = config.get('class_hex_colors', [
        "#ff6347", "#ffa500", "#2ca02c", "#17becf", "#32cd32",
        "#f5deb3", "#8b4513", "#e377c2", "#00008b", "#808080", "#ffffff"
    ])
    
    label_mapping = config.get('label_mapping', {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10
    })
    
    initialize_gpu()
    
    if not image_path:
        print("Error: No image path provided. Please specify an image path in the config or with --image_path.")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    if not os.path.exists(model_path) and not os.path.exists(weights_path):
        print(f"Error: Neither model file ({model_path}) nor weights file ({weights_path}) found.")
        return
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects={'combined_loss': combined_loss})
    else:
        print(f"Creating model architecture and loading weights from {weights_path}")
        img_channels = len(get_satellite_bands(satellite_type))
        
        from tensorflow.keras import Input, Model
        from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Conv2DTranspose, concatenate,
                                           BatchNormalization, LeakyReLU, SpatialDropout2D, Add,
                                           Activation, Multiply, UpSampling2D)
        from tensorflow.keras.regularizers import l2
        
        def attention_block(x, gating, inter_shape):
            theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
            phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating)
            concat = Add()([theta_x, phi_g])
            act = Activation('relu')(concat)
            psi = Conv2D(1, (1, 1), padding='same')(act)
            sigmoid = Activation('sigmoid')(psi)
            upsampled_sigmoid = UpSampling2D(size=(2, 2))(sigmoid)
            y = Multiply()([x, upsampled_sigmoid])
            return y

        def residual_block(x, filters, kernel_size=3, dropout_rate=0.2):
            res = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))(x)
            res = BatchNormalization()(res)
            res = LeakyReLU(alpha=0.1)(res)
            res = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))(res)
            res = BatchNormalization()(res)
            res = LeakyReLU(alpha=0.1)(res)
            res = SpatialDropout2D(dropout_rate)(res)
            return Add()([res, x])
        
        num_classes = len(class_names)
        img_height, img_width = patch_size, patch_size  
        initial_filters = 128
        
        inputs = Input((img_height, img_width, img_channels))
        s = inputs

        c1 = Conv2D(initial_filters, (3, 3), kernel_initializer='he_normal', padding='same')(s)
        c1 = BatchNormalization()(c1)
        c1 = LeakyReLU(alpha=0.1)(c1)
        c1 = SpatialDropout2D(0.2)(c1)
        c1 = residual_block(c1, initial_filters, kernel_size=3, dropout_rate=0.2)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(initial_filters*2, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = LeakyReLU(alpha=0.1)(c2)
        c2 = SpatialDropout2D(0.3)(c2)
        c2 = residual_block(c2, initial_filters*2, kernel_size=3, dropout_rate=0.3)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(initial_filters*4, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = LeakyReLU(alpha=0.1)(c3)
        c3 = SpatialDropout2D(0.4)(c3)
        c3 = residual_block(c3, initial_filters*4, kernel_size=3, dropout_rate=0.4)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(initial_filters*8, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = LeakyReLU(alpha=0.1)(c4)
        c4 = SpatialDropout2D(0.5)(c4)
        c4 = residual_block(c4, initial_filters*8, kernel_size=3, dropout_rate=0.5)
        p4 = MaxPooling2D((2, 2))(c4)

        c5 = Conv2D(initial_filters*16, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
        c5 = BatchNormalization()(c5)
        c5 = LeakyReLU(alpha=0.1)(c5)
        c5 = SpatialDropout2D(0.5)(c5)
        c5 = residual_block(c5, initial_filters*16, kernel_size=3, dropout_rate=0.5)

        u6 = Conv2DTranspose(initial_filters*8, (2, 2), strides=(2, 2), padding='same')(c5)
        c4_att = attention_block(c4, c5, initial_filters*8)
        u6 = concatenate([u6, c4_att])
        c6 = Conv2D(initial_filters*8, (3, 3), kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = LeakyReLU(alpha=0.1)(c6)
        c6 = SpatialDropout2D(0.5)(c6)
        c6 = residual_block(c6, initial_filters*8, kernel_size=3, dropout_rate=0.5)

        u7 = Conv2DTranspose(initial_filters*4, (2, 2), strides=(2, 2), padding='same')(c6)
        c3_att = attention_block(c3, c6, initial_filters*4)
        u7 = concatenate([u7, c3_att])
        c7 = Conv2D(initial_filters*4, (3, 3), kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = LeakyReLU(alpha=0.1)(c7)
        c7 = SpatialDropout2D(0.4)(c7)
        c7 = residual_block(c7, initial_filters*4, kernel_size=3, dropout_rate=0.4)

        u8 = Conv2DTranspose(initial_filters*2, (2, 2), strides=(2, 2), padding='same')(c7)
        c2_att = attention_block(c2, c7, initial_filters*2)
        u8 = concatenate([u8, c2_att])
        c8 = Conv2D(initial_filters*2, (3, 3), kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = LeakyReLU(alpha=0.1)(c8)
        c8 = SpatialDropout2D(0.3)(c8)
        c8 = residual_block(c8, initial_filters*2, kernel_size=3, dropout_rate=0.3)

        u9 = Conv2DTranspose(initial_filters, (2, 2), strides=(2, 2), padding='same')(c8)
        c1_att = attention_block(c1, c8, initial_filters)
        u9 = concatenate([u9, c1_att])
        c9 = Conv2D(initial_filters, (3, 3), kernel_initializer='he_normal', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = LeakyReLU(alpha=0.1)(c9)
        c9 = SpatialDropout2D(0.2)(c9)
        c9 = residual_block(c9, initial_filters, kernel_size=3, dropout_rate=0.2)

        outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy'])
        
        model.load_weights(weights_path)
    
    print(f"Loading image from {image_path} using satellite type {satellite_type}")
    sat_image, height, width, profile, transform = load_tiff_image(image_path, satellite_type)
    
    if sat_image is None:
        print("Failed to load image. Exiting.")
        return
    
    true_label = None
    label_profile = None
    label_transform = None
    
    if label_path and os.path.exists(label_path):
        print(f"Loading ground truth label from {label_path}")
        true_label, label_profile, label_transform = load_label_image(label_path)
        
        if true_label is not None:
            true_label = map_labels(true_label, label_mapping)
            print(f"Ground truth loaded and mapped to {len(class_names)} classes")
    
    print(f"\nStarting prediction with patch size {patch_size}...")
    predicted_label = predict_patches(model, sat_image, patch_size)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nCreating visualization...")
    vis_path = os.path.join(prediction_dir, f"prediction_visualization_{timestamp}.png")
    plot_prediction(sat_image, true_label, predicted_label, class_hex_colors, class_names, vis_path)
    
    print("\nSaving GeoTIFF outputs...")
    save_geotiff(
        predicted_label, 
        f"predicted_labels_{timestamp}.tif", 
        profile if label_profile is None else label_profile, 
        prediction_dir,
        class_hex_colors
    )
    
    rgb_image = normalize_rgb_4channel(sat_image) * 255
    rgb_image = np.transpose(rgb_image, (2, 0, 1)).astype(np.uint8)
    with rasterio.open(os.path.join(prediction_dir, f"satellite_rgb_{timestamp}.tif"), 'w', 
                      driver='GTiff', 
                      height=height, 
                      width=width, 
                      count=3, 
                      dtype='uint8',
                      crs=profile['crs'],
                      transform=transform,
                      compress='lzw') as dst:
        dst.write(rgb_image)
    
    if true_label is not None:
        print("\nCalculating evaluation metrics...")
        metrics_df, cm = calculate_metrics(true_label, predicted_label, class_names)
        
        print("\nEvaluation Results:")
        mean_row = metrics_df[metrics_df['Class'] == 'Mean']
        fw_iou_row = metrics_df[metrics_df['Class'] == 'Frequency Weighted IoU']
        
        print(f"Mean Precision: {mean_row['Precision'].iloc[0]:.4f}")
        print(f"Mean Recall: {mean_row['Recall'].iloc[0]:.4f}")
        print(f"Mean IoU: {mean_row['IoU'].iloc[0]:.4f}")
        print(f"Frequency Weighted IoU: {fw_iou_row['IoU'].iloc[0]:.4f}")
        
        print("\nSaving evaluation results...")
        metrics_path = os.path.join(prediction_dir, f"metrics_{timestamp}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        cm_path = os.path.join(prediction_dir, f"confusion_matrix_{timestamp}.png")
        plot_confusion_matrix(cm, class_names, cm_path)
        
        metrics_plot_path = os.path.join(prediction_dir, f"metrics_plot_{timestamp}.png")
        plot_metrics(metrics_df, metrics_plot_path)
    
    print(f"\nPrediction complete! Results saved to {prediction_dir}")


if __name__ == "__main__":
    main()