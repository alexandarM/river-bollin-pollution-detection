# inference.py
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

from config import DEVICE, IMAGENET_MEAN, IMAGENET_STD
from autoencoder import ConvAutoencoder
from classifier import build_efficientnet
from ensemble import normalize_scores, ensemble_scores


def load_models(ae_path, clf_path, device=DEVICE):
    """Load pre-trained autoencoder and classifier."""
    ae = ConvAutoencoder().to(device)
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()

    clf = build_efficientnet(num_classes=1, pretrained=False).to(device)
    clf.load_state_dict(torch.load(clf_path, map_location=device))
    clf.eval()

    return ae, clf


def preprocess_image(image, for_ae=False):
    # Preprocess PIL image or numpy array for model input
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if for_ae:
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return tf(image).unsqueeze(0)


def predict_single_image(ae, clf, image, device=DEVICE):
    # Run inference on a single image (PIL or numpy)
    # AE error
    ae_input = preprocess_image(image, for_ae=True).to(device)
    with torch.no_grad():
        recon = ae(ae_input)
        mse = torch.nn.functional.mse_loss(recon, ae_input).item()

    # Classifier probability
    clf_input = preprocess_image(image, for_ae=False).to(device)
    with torch.no_grad():
        logit = clf(clf_input).squeeze().item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

    # Ensemble (simplified: we need reference min/max for AE; here we approximate)
    # In practice, there would need to be precomputed min/max from training set.
    # For demo, just return both.
    return {
        'ae_error': mse,
        'clf_prob': prob,
        'ensemble': (mse + prob) / 2  # rough; proper normalization needed
    }


def live_feed(ae, clf, camera_id=0, device=DEVICE):
    """
    Run live inference on webcam feed.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Live feed started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict
        result = predict_single_image(ae, clf, frame, device)
        label = "POLLUTION" if result['ensemble'] > 0.5 else "CLEAN"
        color = (0, 0, 255) if result['ensemble'] > 0.5 else (0, 255, 0)

        # Display
        cv2.putText(frame, f"{label} (AE:{result['ae_error']:.3f} CLF:{result['clf_prob']:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('River Bollin Pollution Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
