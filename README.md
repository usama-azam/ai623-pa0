# AI 623 — Deep Vision Language Models · Assignment 0

Usama Azam · LUMS, Spring 2026

A warm-up tour of the building blocks behind modern vision-language models: deep CNNs, Vision Transformers, variational autoencoders and CLIP. Each task is a self-contained, executed Jupyter notebook in PyTorch.

## Contents

| Notebook | Topic | Dataset |
|---|---|---|
| [`task1-resnet152.ipynb`](task1-resnet152.ipynb) | Inner workings of ResNet-152 | CIFAR-10 |
| [`task2-vit.ipynb`](task2-vit.ipynb) | Understanding Vision Transformers (`google/vit-base-patch16-224`) | ImageNet samples, CIFAR-10 |
| [`task3-vae.ipynb`](task3-vae.ipynb) | Training variational autoencoders | FashionMNIST |
| [`task4-clip.ipynb`](task4-clip.ipynb) | Exploring CLIP | CIFAR-100 |

## Task 1 — ResNet-152

- **Frozen backbone baseline:** pretrained ResNet-152 with a new CIFAR-10 head, training only the head. Validation accuracy **87.45%** after 5 epochs.
- **Skip connections removed** in selected residual blocks. Accuracy collapses to **14.46%** (−73 points), showing how residuals keep gradients flowing in very deep networks.
- **Feature hierarchies:** forward hooks on early, middle and late layers, visualized with t-SNE/UMAP to show class separability emerging with depth.
- **Transfer learning:**

  | Setting | Val. accuracy |
  |---|---|
  | Pretrained + fine-tune final block | 93.05% |
  | Pretrained + fine-tune full backbone | 96.44% |
  | Random initialization + full network | 41.06% |

## Task 2 — Vision Transformers

- Top-1 classification with a pretrained ViT-B/16.
- CLS-token attention maps from the last layer (averaged and per head), reshaped to the 14×14 patch grid and overlaid on the image.
- **Robustness to patch masking:** accuracy stays at 100% with 25% of patches masked, drops to 80% at 50% and 20% at 75%.
- CLS-token vs. mean-pooling representations compared.

## Task 3 — Variational Autoencoders

- Convolutional VAE (20-dim latent) on FashionMNIST, trained with an MSE reconstruction + KL objective. Test loss ≈ 24.3.
- Reconstructions, samples from the prior and latent-space visualizations.
- Posterior collapse investigated and mitigated with **KL annealing** (β warm-up from 0).

## Task 4 — CLIP

- **Zero-shot classification** on CIFAR-100: **63.8%** with no task-specific training.
- **Prompt engineering:** templates range from 56.2% to **65.3%** (best: `"an image of a {}"`).
- Text-to-image and image-to-text retrieval, plus an analysis of the joint embedding space.

## Running

The notebooks were run on a GPU (Kaggle). Main dependencies: `torch`, `torchvision`, `transformers`, `scikit-learn`, `umap-learn`, `matplotlib`. Open a notebook and run all cells; datasets download automatically through `torchvision`.

## License

[MIT](LICENSE)
