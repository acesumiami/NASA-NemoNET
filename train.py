import tensorflow as tf
import numpy as np
import os
import json
import yaml
import cv2
import rasterio
from rasterio.enums import Resampling
import argparse
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Conv2DTranspose, concatenate,
                                    BatchNormalization, LeakyReLU, SpatialDropout2D, Add,
                                    Activation, Multiply, UpSampling2D)
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2


def parse_args():
    parser = argparse.ArgumentParser(description='Train a U-Net model for satellite image segmentation')
    parser.add_argument('--config_path', type=str, required=True, help='Path to YAML configuration file')
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


class NeMOAugmentationModule:
    def __init__(self, num_channels=4, pixel_mean=None, pixel_std=None, reverse_normalize=False):
        self.num_channels = num_channels
        self.pixel_mean = self._apply_channel_corrections(pixel_mean, num_channels)
        self.pixel_std = self._apply_channel_corrections(pixel_std, num_channels)
        self.reverse_normalize = reverse_normalize

    def _apply_channel_corrections(self, value, num_channels):
        if value is None:
            return [0.0] * num_channels
        elif isinstance(value, (float, int)):
            return [value] * num_channels
        elif len(value) != num_channels:
            raise ValueError(f"Channel correction length {len(value)} does not match channels: {num_channels}")
        return value

    def _normalize(self, input_array, reverse=False):
        if reverse:
            return (input_array * self.pixel_std) + self.pixel_mean
        else:
            return (input_array - self.pixel_mean) / self.pixel_std

    @staticmethod
    def random_flip_rotation(input_image, rnd_flip=True, rnd_rotation=True):
        x = input_image
        flip = 0
        num_rotate = 0
        if rnd_flip:
            flip = np.random.randint(0, 2)
        if rnd_rotation:
            num_rotate = np.random.randint(0, 4)
        if flip:
            x = np.flip(x, axis=0)
        x = np.rot90(x, k=num_rotate)
        return x, flip, num_rotate

    @staticmethod
    def flip_rotation(input_image, flip=0, rotation=0):
        x = input_image
        if flip:
            x = np.flip(x, axis=0)
        x = np.rot90(x, k=rotation)
        return x


class PolynomialAugmentation(NeMOAugmentationModule):
    def __init__(self, num_channels=4, pixel_mean=None, pixel_std=None, reverse_normalize=True):
        super().__init__(num_channels, pixel_mean, pixel_std, reverse_normalize)
        self.fit = {}
        self.fit[1] = np.asarray([
            [-0.01142, 1.74007, -0.000742, -0.22785, 0.00781, -0.23779, 0.0, 0.0, 26.4154],
            [0.00091, -1.46364, -0.00491, 0.14658, 0.00149, 1.63454, 0.0, 0.0, 85.6772],
            [0.00748, -2.45993, -0.00375, 0.32433, -0.00239, 1.55765, 0.0, 0.0, 105.2049],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0525, 2.22991, 4.8516]
        ])

    def apply(self, input_image):
        x = self._normalize(input_image, self.reverse_normalize)
        x = np.rollaxis(x, 2, 0)
        white_pixels = np.where(np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) >= 250)
        NIR_pixels = np.where(x[3] >= 50)
        toremap = np.random.randint(0, len(self.fit) + 1)
        xorig = np.copy(x)
        if toremap > 0:
            for i in range(self.num_channels):
                x[i] = self.apply_polynomial_fit2channel(xorig, self.fit[toremap], i)
                x[i][white_pixels] = xorig[i][white_pixels]
                x[i][NIR_pixels] = xorig[i][NIR_pixels]
        x = np.rollaxis(x, 0, 3)
        x = self._normalize(x, not self.reverse_normalize)
        return x

    @staticmethod
    def apply_polynomial_fit2channel(x, p, idx):
        channel = (p[idx, 0]*x[0]**2 + p[idx, 1]*x[0] +
                  p[idx, 2]*x[1]**2 + p[idx, 3]*x[1] +
                  p[idx, 4]*x[2]**2 + p[idx, 5]*x[2] +
                  p[idx, 6]*x[3]**2 + p[idx, 7]*x[3] +
                  p[idx, 8] + np.random.uniform(-0.1, 0.1, x[0].shape))
        return channel


def get_satellite_bands(satellite_type):
    if satellite_type.lower() == 'wv2':
        return [2, 3, 5, 7]  # WorldView-2: Blue, Green, Red, NIR1
    else:  # Default to Sentinel-2
        return [1, 2, 3, 4]  # Sentinel-2: Blue, Green, Red, NIR


def load_tiff_image(image_path, img_height, img_width, satellite_type='sn2'):
    try:
        bands = get_satellite_bands(satellite_type)
        with rasterio.open(image_path) as dataset:
            image = dataset.read(bands, out_shape=(len(bands), img_height, img_width), 
                                resampling=Resampling.bilinear)
            image = np.transpose(image, (1, 2, 0))
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return np.zeros((img_height, img_width, len(get_satellite_bands(satellite_type))), dtype=np.float32)


def load_label_image(label_path, img_height, img_width):
    try:
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Unable to open {label_path}")
        if label.shape[:2] != (img_height, img_width):
            label = cv2.resize(label, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        return label
    except Exception as e:
        print(f"Error loading label {label_path}: {e}")
        return np.zeros((img_height, img_width), dtype=np.uint8)


def load_images_and_labels(image_dir, label_dir, img_height, img_width, satellite_type='sn2'):
    images, labels = [], []
    print(f"Loading images and labels from {image_dir} and {label_dir}")
    for class_name in os.listdir(image_dir):
        class_image_dir = os.path.join(image_dir, class_name)
        class_label_dir = os.path.join(label_dir, class_name)
        if os.path.isdir(class_image_dir):
            for image_name in os.listdir(class_image_dir):
                image_path = os.path.join(class_image_dir, image_name)
                label_name = image_name.replace('.tif', '.png')
                label_path = os.path.join(class_label_dir, label_name)
                try:
                    images.append(load_tiff_image(image_path, img_height, img_width, satellite_type))
                    labels.append(load_label_image(label_path, img_height, img_width))
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
    print(f"Loaded {len(images)} images and {len(labels)} labels")
    return np.array(images), np.array(labels)


def map_labels(labels, mapping, default=9):
    vectorized_map = np.vectorize(lambda x: mapping.get(x, default))
    return vectorized_map(labels)


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


def complex_unet_model(n_classes, img_height, img_width, img_channels, initial_filters=128):
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

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def normalize_rgb_4channel(image, lower_percentile=2, upper_percentile=98):
    if image.shape[-1] < 3:
        raise ValueError("Image does not have at least 3 channels.")
    elif image.shape[-1] == 3:
        rgb_image = image
    elif image.shape[-1] >= 4:
        rgb_image = image[:, :, :3]
    else:
        raise ValueError("Unsupported number of channels.")

    image_display = np.zeros_like(rgb_image, dtype=np.float32)
    for c in range(3):
        channel = rgb_image[:, :, c]
        p_low = np.percentile(channel, lower_percentile)
        p_high = np.percentile(channel, upper_percentile)
        image_display[:, :, c] = np.clip(
            (channel - p_low) / (p_high - p_low + 1e-8), 0, 1)
    return image_display


def main():
    args = parse_args()
    
    config = load_config(args.config_path)
    
    satellite_type = args.satellite if args.satellite else config.get('satellite_type', 'sn2')
    
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    coral_classes_config = config['paths'].get('coral_classes_config')
    if coral_classes_config and os.path.exists(coral_classes_config):
        try:
            with open(coral_classes_config) as json_file:
                coral_classes = json.load(json_file)
            print("Coral Classes loaded successfully.")
        except Exception as e:
            print(f"Error loading Coral Classes configuration: {e}")
            coral_classes = None
    
    img_height = config['image_set_loader']['train']['image_size'][0]
    img_width = config['image_set_loader']['train']['image_size'][1]
    img_channels = len(get_satellite_bands(satellite_type))
    
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
    
    num_classes = len(class_names)
    
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = config['training']['optimizer']['learning_rate']
    
    print(f"\nStarting training with the following configuration:")
    print(f"Satellite type: {satellite_type}")
    print(f"Image dimensions: {img_height}x{img_width}x{img_channels}")
    print(f"Number of classes: {num_classes}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {epochs}")
    print(f"Output directory: {output_dir}")
    
    initialize_gpu()
    
    train_image_dir = config['image_set_loader']['train']['image_dir']
    train_label_dir = config['image_set_loader']['train']['label_dir']
    val_image_dir = config['image_set_loader']['val']['image_dir']
    val_label_dir = config['image_set_loader']['val']['label_dir']
    
    X_train, y_train = load_images_and_labels(train_image_dir, train_label_dir, 
                                             img_height, img_width, satellite_type)
    X_val, y_val = load_images_and_labels(val_image_dir, val_label_dir, 
                                         img_height, img_width, satellite_type)
    
    augmentor = PolynomialAugmentation(num_channels=img_channels, 
                                      pixel_mean=[100]*img_channels, 
                                      pixel_std=[100]*img_channels, 
                                      reverse_normalize=True)
    X_train_aug = np.array([augmentor.apply(img) for img in X_train])
    
    X_train_combined = np.concatenate((X_train, X_train_aug))
    y_train_combined = np.concatenate((y_train, y_train))
    
    y_train_combined = map_labels(y_train_combined, label_mapping).astype(np.uint8)
    y_val = map_labels(y_val, label_mapping).astype(np.uint8)
    
    X_train_combined = X_train_combined.astype('float32')
    X_val = X_val.astype('float32')
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_combined, y_train_combined)) \
        .shuffle(buffer_size=1000) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
        .batch(batch_size) \
        .prefetch(tf.data.AUTOTUNE)
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        print(f"Using {strategy.num_replicas_in_sync} device(s) for training")
        model = complex_unet_model(num_classes, img_height, img_width, img_channels)
        model.summary()
    
    mixed_precision.set_global_policy('mixed_float16')
    
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=['accuracy']
        )
    
    callbacks = [
        EarlyStopping(
            monitor=config['training']['callbacks']['early_stopping']['monitor'],
            patience=config['training']['callbacks']['early_stopping']['patience'],
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor=config['training']['callbacks']['reduce_lr']['monitor'],
            factor=config['training']['callbacks']['reduce_lr']['factor'],
            patience=config['training']['callbacks']['reduce_lr']['patience']
        )
    ]
    
    print("\nStarting model training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()
    
    model_save_path = os.path.join(output_dir, "model.h5")
    weights_save_path = os.path.join(output_dir, "weights.h5")
    
    print("\nSaving model and weights...")
    model.save(model_save_path)
    model.save_weights(weights_save_path)
    
    config_with_results = config.copy()
    config_with_results['training_results'] = {
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
    }
    config_with_results['satellite_type'] = satellite_type
    
    with open(os.path.join(output_dir, 'training_config.yaml'), 'w') as f:
        yaml.dump(config_with_results, f, default_flow_style=False)
    
    print(f"\nTraining complete! Model saved to {model_save_path}")
    print(f"Training configuration and results saved to {os.path.join(output_dir, 'training_config.yaml')}")


if __name__ == "__main__":
    main()
