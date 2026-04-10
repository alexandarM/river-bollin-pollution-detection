# River Bollin Pollution Detection

This is a machine learning project I built to detect pollution incidents in the River Bollin, Macclesfield, UK using images from a live webcam. The idea is that the model runs continuously on the webcam feed and triggers an alert whenever it detects something that looks like a pollution event, so that authorities can be notified as quickly as possible.

---

## Dataset and Data Provider

The dataset was collected and released by **MarkP1929** on HuggingFace: [MarkP1929/mill-st-imgs](https://huggingface.co/datasets/MarkP1929/mill-st-imgs).

The images come from a real webcam overlooking the River Bollin and were scraped and stored over roughly 8 months. The dataset contains around 50,000 images in total, but only about 50 of them are actual pollution incidents. This makes it a highly imbalanced dataset, which is a typical challenge in real-world scenarios like this one.

The dataset includes:
- An `imgs/` folder with all 224x224 JPG images
- A `pollution_incidents.txt` file listing the filenames of confirmed pollution events
- Total download size is approximately 600 MB

The dataset was collected and released by MarkP1929, who spent around 8 months scraping webcam images of the river and labelling pollution incidents. Thanks also to the Friends of the River Bollin for their involvement. The River Bollin has a recurring pollution problem and the motivation behind this project is to speed up detection and response time. Salmon and Otters have been spotted in the river downstream, which makes it worth protecting.

---

## The Problem

With 50 positives out of 50,000 images the class imbalance is roughly 1:1000. A model that just predicts "clean" for every single image would achieve 99.9% accuracy, which is completely useless. Because of this, accuracy is not the right metric here at all. I used PR-AUC (Precision-Recall AUC) instead, which actually reflects how well the model finds the rare positive cases.

---

## Approach

I combined two models into an ensemble because neither one alone handles the imbalance perfectly.

**Convolutional Autoencoder (anomaly detection)**

The autoencoder is trained only on clean images, so it learns what a normal river frame looks like. When a pollution event happens (unusual colour, foam, discolouration) the model struggles to reconstruct it accurately, which shows up as a high MSE reconstruction error. Anything above a calibrated threshold gets flagged.

**EfficientNet-B0 (supervised classifier)**

A pretrained EfficientNet-B0 with a custom binary output head, fine-tuned using Focal Loss (alpha=0.75, gamma=2.0) to handle the class imbalance. A WeightedRandomSampler is used during training so that each batch sees roughly 50% pollution images, preventing the model from collapsing to always predicting clean.

**Ensemble**

The AE reconstruction error and classifier probability are both normalised to [0,1] and averaged. If either model flags a frame, an alert is triggered.

---

## Repository Structure

The project uses a flat file structure:

```
river-bollin-pollution/
│
├── river_bollin_pollution_final.ipynb   # main Colab notebook
├── config.py                            # seeds, hyperparameters, device
├── dataset_utils.py                     # dataset classes, transforms, sampler
├── focal_loss.py                        # focal loss implementation
├── autoencoder.py                       # ConvAutoencoder model definition
├── train_autoencoder.py                 # autoencoder training loop
├── classifier.py                        # EfficientNet build, train, evaluate
├── ensemble.py                          # ensemble scoring and evaluation
├── inference.py                         # load models, single image prediction
├── visualization.py                     # plotting utilities
├── live_monitor.py                      # live webcam monitoring loop
└── README.md
```

---

## How to Run

### Google Colab (recommended)

1. Open `river_bollin_pollution_final.ipynb` in Google Colab
2. Set the runtime to T4 GPU under Runtime > Change runtime type
3. Add your HuggingFace token to Colab Secrets as `HF_TOKEN` (needed to download the dataset)
4. Run cells from top to bottom

The notebook handles everything: downloading the dataset, training both models, evaluating them, and running the live monitor.

### Requirements

```
torch
torchvision
scikit-learn
matplotlib
pillow
tqdm
huggingface_hub
requests
pandas
```

Install with:

```bash
pip install torch torchvision scikit-learn matplotlib pillow tqdm huggingface_hub requests pandas
```

---

## Results

| Model | PR-AUC |
|-------|--------|
| Autoencoder | 0.0064 |
| EfficientNet-B0 | 1.0 |
| Ensemble | 0.9924 |

The classifier PR-AUC of 1.0 on the test set should be interpreted carefully given the very small number of positives (roughly 7-8 in the test split). The ensemble result of 0.9924 is more representative of real-world performance. The autoencoder on its own scores poorly in PR-AUC terms but still contributes to the ensemble by catching visual anomalies the classifier might miss.

---

## Live Monitoring

Once the models are trained, the live monitor fetches a new frame from the webcam every 60 seconds, runs inference, and saves the result to Google Drive:

```
bollin_pollution/
├── images/
│   ├── clean/       img_2026-03-14_14-30_clean.jpg
│   ├── pollution/   img_2026-03-14_15-45_POLLUTION.jpg
│   └── dark/        img_2026-03-14_02-00_dark.jpg
├── monitor_log.csv
└── alerts.log
```

Dark frames (night-time, mean brightness below threshold) are skipped and saved separately since you cannot detect pollution in an image where nothing is visible. Duplicate frames are detected via MD5 hash and skipped so that repeated fetches before the webcam refreshes do not fill up storage.

The webcam feed used is: http://www.maccinfo.com/WebcamUtility/Utility.jpg — it updates roughly every minute.

---

## Limitations

- The autoencoder threshold needs to be recalibrated periodically as lighting and seasonal conditions change. A threshold that works in summer may not work in winter.
- With only around 50 confirmed positive examples the classifier may not generalise well to pollution types it has not seen before. More labelled examples over time will improve this.
- Colab free tier disconnects after roughly 90 minutes of inactivity, which interrupts the live monitor. Colab Pro or a persistent VM would be needed for proper 24/7 deployment.
- Night-time frames are skipped entirely, meaning pollution events that happen after dark will not be caught by the current system.

---

## Acknowledgements

Dataset collected and provided by MarkP1929 via HuggingFace. Thanks also to the Friends of the River Bollin for their environmental monitoring work that made this dataset possible.
