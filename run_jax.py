#!/usr/bin/env python3

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from typing import Any, Callable, Sequence
import matplotlib
matplotlib.use('Agg')
import time
import psutil
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import gc
import augmax
import csv
from datetime import datetime

print("=" * 70)
print("JAX Brain Tumor Detection")
print("=" * 70)

def get_cpu_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Returns used memory in MB
        return info.used / 1024 / 1024
    except ImportError:
        # Fallback if pynvml is not installed: read from nvidia-smi via shell
        try:
            import subprocess
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
                encoding='utf-8'
            )
            return float(result.strip())
        except:
            return 0.0
    except Exception:
        return 0.0

print(f"\nJAX version: {jax.__version__}")
devices = jax.devices()
gpu_devices = [d for d in devices if d.platform == 'gpu']

if len(gpu_devices) > 0:
    print(f"GPU detected: {gpu_devices}")
    default_device = gpu_devices[0]
else:
    print("WARNING: No GPU detected! Running on CPU.")

data_dir = Path("./data/brain_tumor_dataset")
if not data_dir.exists():
    raise FileNotFoundError(f"Dataset not found at {data_dir}")

class_names = ['No Tumor', 'Tumor']

def load_image(path, target_size=(128, 128)):
    img = Image.open(path).convert('RGB')
    img = img.resize(target_size, Image.BILINEAR)
    img = np.array(img) / 255.0
    return img.astype(np.float32)

def normalize_batch_jax(batch_images):
    mean = jnp.array([0.248, 0.248, 0.249]).reshape(1, 1, 1, 3)
    std = jnp.array([0.211, 0.211, 0.211]).reshape(1, 1, 1, 3)
    return (batch_images - mean) / std

augment_transform = augmax.Chain(
    augmax.HorizontalFlip(p=0.5),
    augmax.VerticalFlip(p=0.3),
    augmax.Rotate((-20, 20)),
    augmax.Warp(strength=0.001, coarseness=32),
    augmax.GaussianBlur(sigma=1.0),
    augmax.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
)

image_paths = []
labels = []

for class_idx, class_name in enumerate(['no', 'yes']):
    class_dir = data_dir / class_name
    for img_path in class_dir.glob('*.jpg'):
        image_paths.append(str(img_path))
        labels.append(class_idx)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.3, stratify=labels, random_state=42
)

print(f"\nDataset Loaded:")
print(f"  Training: {len(train_paths)} samples")
print(f"  Validation: {len(val_paths)} samples")

print("\nLoading images into memory...")
train_images = np.array([load_image(path) for path in train_paths])
val_images = np.array([load_image(path) for path in val_paths])
train_labels_array = np.array(train_labels, dtype=np.int32)
val_labels_array = np.array(val_labels, dtype=np.int32)

batch_size = 32
num_classes = len(class_names)

class BrainTumorCNN(nn.Module):
    num_classes: int = 2

    @nn.compact
    def __call__(self, x, training: bool = True):
        init = nn.initializers.he_uniform()

        x = nn.Conv(32, (3, 3), padding='SAME', kernel_init=init)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(64, (3, 3), padding='SAME', kernel_init=init)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=0.3, deterministic=not training)(x)
        
        x = nn.Conv(128, (3, 3), padding='SAME', kernel_init=init)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(128, (3, 3), padding='SAME', kernel_init=init)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=0.3, deterministic=not training)(x)
        
        x = x.reshape((x.shape[0], -1))
        
        x = nn.Dense(512, kernel_init=init)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        
        x = nn.Dense(128, kernel_init=init)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        
        x = nn.Dense(self.num_classes, kernel_init=init)(x)
        return x

model = BrainTumorCNN(num_classes=num_classes)
key = random.PRNGKey(42)
key, init_key = random.split(key)
dummy_input = jnp.ones((1, 128, 128, 3))
params = model.init(init_key, dummy_input, training=False)

print(f"Model parameters initialized on {default_device}")

class TrainState(train_state.TrainState):
    batch_stats: Any = None

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=num_classes)
    loss = optax.softmax_cross_entropy(logits, one_hot)
    return jnp.mean(loss)

def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}

@jit
def train_step(state, batch_images, batch_labels, rng_key):
    aug_key, dropout_key, noise_key = random.split(rng_key, 3)
    batch_keys = random.split(aug_key, batch_images.shape[0])
    batch_images = jax.vmap(augment_transform)(batch_keys, batch_images)
    noise = random.normal(noise_key, batch_images.shape) * 0.02
    batch_images = batch_images + noise
    batch_images = normalize_batch_jax(batch_images)

    def loss_fn(params):
        logits = model.apply(params, batch_images, training=True, rngs={'dropout': dropout_key})
        loss = cross_entropy_loss(logits, batch_labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, batch_labels)
    return state, metrics

@jit
def eval_step(params, batch_images, batch_labels):
    batch_images = normalize_batch_jax(batch_images)
    logits = model.apply(params, batch_images, training=False)
    return compute_metrics(logits, batch_labels), logits

def create_batches(images, labels, batch_size, shuffle=False, key=None):
    num_samples = len(images)
    indices = np.arange(num_samples)
    if shuffle and key is not None:
        key, shuffle_key = random.split(key)
        indices = random.permutation(shuffle_key, indices)
        indices = np.array(indices)
    for i in range(0, num_samples, batch_size):
        idx = indices[i:i+batch_size]
        yield images[idx], labels[idx], key

def update_learning_rate(state, new_lr):
    new_hparams = state.opt_state.hyperparams.copy()
    new_hparams['learning_rate'] = new_lr
    new_opt_state = state.opt_state._replace(hyperparams=new_hparams)
    return state.replace(opt_state=new_opt_state)

def train_model(params, num_epochs=150):
    initial_learning_rate = 0.001
    min_lr = 1e-5
    current_lr = initial_learning_rate
    optimizer = optax.inject_hyperparams(optax.adamw)(
        learning_rate=current_lr,
        weight_decay=0.01
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    epoch_times = []
    
    best_val_loss = float('inf')
    best_params = None
    best_epoch = 0
    
    scheduler_patience = 20
    scheduler_factor = 0.7
    scheduler_counter = 0
    scheduler_best_loss = float('inf')
    
    # Init tracking
    max_cpu_memory = get_cpu_memory()
    max_gpu_memory = get_gpu_memory()
    
    total_start_time = time.time()
    loop_key = random.PRNGKey(999)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        loop_key, train_key = random.split(loop_key)
        
        # Train Loop
        batch_metrics = []
        for img, lbl, step_key in create_batches(
            train_images, train_labels_array, batch_size, shuffle=True, key=train_key
        ):
            state, m = train_step(state, img, lbl, step_key)
            batch_metrics.append(m)
        
        train_loss = np.mean([m['loss'] for m in batch_metrics])
        train_acc = np.mean([m['accuracy'] for m in batch_metrics]) * 100
        
        # Validation Loop
        batch_metrics = []
        for img, lbl, _ in create_batches(
            val_images, val_labels_array, batch_size, shuffle=False
        ):
            m, _ = eval_step(state.params, img, lbl)
            batch_metrics.append(m)
        
        val_loss = np.mean([m['loss'] for m in batch_metrics])
        val_acc = np.mean([m['accuracy'] for m in batch_metrics]) * 100
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Update Peaks
        curr_cpu = get_cpu_memory()
        curr_gpu = get_gpu_memory()
        if curr_cpu > max_cpu_memory: max_cpu_memory = curr_cpu
        if curr_gpu > max_gpu_memory: max_gpu_memory = curr_gpu
            
        train_losses.append(float(train_loss))
        train_accs.append(float(train_acc))
        val_losses.append(float(val_loss))
        val_accs.append(float(val_acc))
        
        lr_msg = ""
        if val_loss < scheduler_best_loss:
            scheduler_best_loss = val_loss
            scheduler_counter = 0
        else:
            scheduler_counter += 1
            if scheduler_counter >= scheduler_patience:
                new_lr = max(current_lr * scheduler_factor, min_lr)
                if new_lr != current_lr:
                    current_lr = new_lr
                    state = update_learning_rate(state, current_lr)
                    lr_msg = f" [LR -> {current_lr:.6f}]"
                scheduler_counter = 0
        
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), state.params)
            marker = " <- BEST"
        
        print(f"Epoch {epoch+1}/{num_epochs}{lr_msg}")
        print(f"  Train: Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
        print(f"  Val:   Loss {val_loss:.4f} | Acc {val_acc:.2f}%{marker}")
        print(f"  Time:  {epoch_time:.2f}s")
    
    total_time = time.time() - total_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    print(f"\n{'='*70}")
    print(f"Best Model (Epoch {best_epoch+1})")
    print(f"Val Loss: {best_val_loss:.4f} | Val Acc: {val_accs[best_epoch]:.2f}%")
    print(f"{'='*70}")
    
    return {
        'best_params': best_params,
        'best_epoch': best_epoch,
        'val_accs': val_accs,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'train_losses': train_losses,
        'total_time': total_time,
        'avg_epoch_time': avg_epoch_time,
        'max_cpu_memory': max_cpu_memory,
        'max_gpu_memory': max_gpu_memory
    }

print("\nStarting Training...")
def evaluate_model(params):
    from sklearn.metrics import roc_auc_score
    print("\nEvaluating Best Model...")
    
    eval_start_time = time.time()
    
    all_probs = []
    all_targs = []
    
    for img, lbl, _ in create_batches(val_images, val_labels_array, batch_size, shuffle=False):
        _, logits = eval_step(params, img, lbl)
        probs = jax.nn.softmax(logits)
        all_probs.extend(np.array(probs))
        all_targs.extend(np.array(lbl))
    
    eval_time = time.time() - eval_start_time
    
    tumor_probs = [p[1] for p in all_probs]
    auc = roc_auc_score(all_targs, tumor_probs)
    print(f"Final Test AUC: {auc:.4f}")
    return auc, eval_time

import csv
from datetime import datetime

results_file = Path("results_jax.csv")
file_exists = results_file.exists()

for run_idx in range(10):
    print(f"\n========== RUN {run_idx+1}/10 ==========")
    # Re-initialize model parameters for each run
    key = random.PRNGKey(42)
    key, init_key = random.split(key)
    params = model.init(init_key, dummy_input, training=False)
    history = train_model(params, num_epochs=150)
    auc_score, eval_time = evaluate_model(history['best_params'])

    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists and run_idx == 0:
            writer.writerow([
                'timestamp', 'notebook', 'best_epoch', 
                'train_accuracy', 'val_accuracy', 'auc_score',
                'train_loss', 'val_loss',
                'total_time_minutes', 'avg_epoch_time_seconds',
                'max_cpu_memory_mb', 'max_gpu_memory_mb', 'eval_time_seconds'
            ])
        writer.writerow([
            datetime.now().isoformat(),
            'jax',
            history['best_epoch'] + 1,
            history['train_accs'][history['best_epoch']],
            history['val_accs'][history['best_epoch']],
            auc_score,
            history['train_losses'][history['best_epoch']],
            history['val_losses'][history['best_epoch']],
            history['total_time'] / 60,
            history['avg_epoch_time'],
            history['max_cpu_memory'],
            history['max_gpu_memory'],
            eval_time
        ])

print(f"\n Training Complete. Results saved to {results_file}")