import os
import json
import matplotlib.pyplot as plt

models = ['ResNet50', 'ViT-S16']
pretrain_types = {
    'w pretrain': 'w Pretrain',
    'wo pretrain': 'w/o Pretrain'
}
augmentations = {
    '': 'None',
    'GaussianBlur': 'GaussianBlur'
}

log_files = []
for model in models:
    for pretrain_folder, pretrain_label in pretrain_types.items():
        base_path = os.path.join('logs', model, pretrain_folder)
        path_none = os.path.join(base_path, 'training_log.json')
        if os.path.exists(path_none):
            log_files.append((path_none, model, pretrain_label, 'None'))
        path_gb = os.path.join(base_path, 'GaussianBlur', 'training_log.json')
        if os.path.exists(path_gb):
            log_files.append((path_gb, model, pretrain_label, 'GaussianBlur'))

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
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
    ax.set_title(f'{model} ({pretrain}, Aug: {augmentation})')
    ax.set_ylim(0, 100)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric')
    ax.legend()
    ax.grid(True)

for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("results/train_results.png", dpi=300)