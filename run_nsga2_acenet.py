"""CIFAKE CNN hyperparameter optimization: validation error vs SHAP rank instability.

Usage:
    python run_nsga2_acenet.py --dataset-path /path/to/cifake_dataset

Prepare the dataset separately on a machine with internet access:
    from datasets import load_dataset
    load_dataset('dragonintelligence/CIFAKE-image-dataset').save_to_disk('cifake_dataset')
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from datasets import load_from_disk
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data[index]
        return self.transform(record['image'].convert('RGB')), int(record['label'])


class AIImageDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 8 * 8, 32), nn.ReLU(), nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_data(dataset_path):
    dataset = load_from_disk(str(dataset_path))
    train_data = ImageDataset(dataset['train'])
    indices = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(42))
    small_train = Subset(train_data, indices[:2000].tolist())
    validation_data = Subset(train_data, indices[2000:2200].tolist())
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=False)
    background = torch.stack([small_train[i][0] for i in range(20)])
    return small_train, validation_data, validation_loader, background


def evaluate_cnn(learning_rate, weight_decay, small_train, validation_loader):
    torch.manual_seed(42)
    model = AIImageDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    # Same minibatch order for each candidate, making comparisons fairer.
    train_loader = DataLoader(small_train, batch_size=32, shuffle=True,
                              generator=torch.Generator().manual_seed(42))
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in validation_loader:
            predicted = model(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total, model


def calculate_stability(model, validation_data, background):
    model.eval()
    explainer = shap.GradientExplainer(model, background)
    scores = []
    for i in range(3):
        image, _ = validation_data[i]
        original = image.unsqueeze(0)
        generator = torch.Generator().manual_seed(42 + i)
        noise = torch.randn(original.shape, generator=generator) * 0.02
        perturbed = torch.clamp(original + noise, 0, 1)
        with torch.no_grad():
            output_class = model(original).argmax(dim=1).item()
        original_shap = explainer.shap_values(original, nsamples=100, rseed=42)
        perturbed_shap = explainer.shap_values(perturbed, nsamples=100, rseed=42)
        original_map = original_shap[0, :, :, :, output_class].sum(axis=0)
        perturbed_map = perturbed_shap[0, :, :, :, output_class].sum(axis=0)
        score, _ = spearmanr(np.abs(original_map).ravel(), np.abs(perturbed_map).ravel())
        if not np.isfinite(score):
            raise ValueError('SHAP stability is undefined for a candidate; inspect attributions.')
        scores.append(score)
    return float(np.mean(scores))


class CNNOptimizationProblem(ElementwiseProblem):
    def __init__(self, small_train, validation_data, validation_loader, background):
        super().__init__(n_var=2, n_obj=2, xl=np.array([-4., -6.]), xu=np.array([-2., -2.]))
        self.small_train = small_train
        self.validation_data = validation_data
        self.validation_loader = validation_loader
        self.background = background

    def _evaluate(self, x, out, *args, **kwargs):
        learning_rate, weight_decay = 10 ** x[0], 10 ** x[1]
        accuracy, model = evaluate_cnn(learning_rate, weight_decay,
                                       self.small_train, self.validation_loader)
        stability = calculate_stability(model, self.validation_data, self.background)
        out['F'] = [1 - accuracy, 1 - stability]
        print(f'lr={learning_rate:.6g} wd={weight_decay:.6g} '
              f'accuracy={accuracy:.3f} stability={stability:.3f}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', required=True, type=Path)
    parser.add_argument('--output', default='nsga2_results.csv')
    parser.add_argument('--pop-size', type=int, default=6)
    parser.add_argument('--generations', type=int, default=3)
    args = parser.parse_args()
    small_train, validation_data, validation_loader, background = make_data(args.dataset_path)
    problem = CNNOptimizationProblem(small_train, validation_data, validation_loader, background)
    result = minimize(problem, NSGA2(pop_size=args.pop_size),
                      termination=('n_gen', args.generations), seed=42, verbose=True)
    if result.X is None or result.F is None:
        raise RuntimeError('NSGA-II returned no feasible Pareto solutions.')
    table = pd.DataFrame({
        'Learning_Rate': 10 ** result.X[:, 0],
        'Weight_Decay': 10 ** result.X[:, 1],
        'Accuracy': 1 - result.F[:, 0],
        'SHAP_Stability': 1 - result.F[:, 1],
    })
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
