import os
import json
import matplotlib.pyplot as plt

models = ['ResNet50', 'ViT-S16']
pretrain_types = {
    'w pretrain': 'w Pretrain',
    'wo pretrain': 'w/o Pretrain'
}
augmentations = {
    'None': 'None',
    'GaussianBlur': 'GaussianBlur',
    'RandomErasing': 'RandomErasing'
}

log_files = []
for model in models:
    for pretrain_folder, pretrain_label in pretrain_types.items():
        base_path = os.path.join('logs', model, pretrain_folder)
        
        for aug_folder, aug_label in augmentations.items():
            aug_path = os.path.join(base_path, aug_folder, 'training_log.json')
            if os.path.exists(aug_path):
                log_files.append((aug_path, model, pretrain_label, aug_label))

# Create 2x6 subplot layout
fig, axes = plt.subplots(2, 6, figsize=(30, 10))
axes = axes.flatten()

for idx, (file_path, model, pretrain, augmentation) in enumerate(log_files):
    with open(file_path, 'r') as f:
        log_data = json.load(f)
    epochs = [entry['epoch'] for entry in log_data]
    acc = [entry['accuracy'] for entry in log_data]
    loss = [entry['loss'] for entry in log_data]

    ax = axes[idx]
    ax.plot(epochs, acc, label='Accuracy')
    ax.plot(epochs, loss, label='Loss')
    ax.set_ylim(0, 100)
    ax.set_title(f"{model}\n{pretrain}\n{augmentation}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('results/training_results.png')
plt.close()