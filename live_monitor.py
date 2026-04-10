# live_monitor.py
import io
import time
import hashlib
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import clear_output

from dataset_utils import AE_TF, BASE_TF
from config import DEVICE


def is_too_dark(img: Image.Image, darkness_threshold: float = 30.0) -> bool:
    """Check if image is too dark (mean pixel value < threshold)."""
    gray = img.convert('L')
    return gray.resize((1, 1)).getpixel((0, 0)) < darkness_threshold


def fetch_frame(url: str, timeout: int = 15) -> Image.Image | None:
    # Get image from webcam URL
    try:
        resp = requests.get(
            url, timeout=timeout,
            headers={'Cache-Control': 'no-cache', 'Pragma': 'no-cache'}
        )
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert('RGB')
    except Exception as e:
        print(f'Fetch failed: {e}')
        return None


def image_hash(img: Image.Image) -> str:
    #Compute hash of image for duplicate detection
    return hashlib.md5(img.tobytes()).hexdigest()


def save_frame(img: Image.Image, folder: Path, label: str, ts: datetime) -> Path:
    # Save image to folder with timestamp label
    fname = f"img_{ts.strftime('%Y-%m-%d_%H-%M')}_{label}.jpg"
    path = folder / fname
    img.save(path, format='JPEG', quality=92)
    return path


def log_result(row: dict, log_path: Path):
    #Append result row to CSV log
    df_new = pd.DataFrame([row])
    if log_path.exists():
        pd.concat([pd.read_csv(log_path), df_new], ignore_index=True).to_csv(log_path, index=False)
    else:
        df_new.to_csv(log_path, index=False)


def trigger_alert(ts: datetime, result: dict, saved_path: Path, alert_log: Path):
    """Log alert to file and print."""
    msg = (f"POLLUTION ALERT | {ts.isoformat()} | "
           f"ae_error={result['ae_error']} | "
           f"clf_prob={result['clf_prob']} | "
           f"saved={saved_path.name}")
    print(msg)
    with open(alert_log, 'a') as f:
        f.write(msg + '\n')



def display_status(img: Image.Image, result: dict, ts: datetime,
                   stats: dict, saved_path: Path,
                   ae_threshold: float, clf_threshold: float):
    #Clear output and display live status with plots
    clear_output(wait=True)

    verdict = result['verdict']
    colour = ('red'   if result['pollution'] is True  else
              'green' if result['pollution'] is False else
              'gray')

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1], figure=fig)

    # panel 1: webcam frame
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img)
    ax0.set_title(f'{verdict}\n{ts.strftime("%Y-%m-%d   %H:%M")}',
                  color=colour, fontsize=12, fontweight='bold')
    ax0.axis('off')

    # panel 2: model scores
    ax1 = fig.add_subplot(gs[1])
    if verdict == 'SKIPPED':
        ax1.text(0.5, 0.5, 'Too dark\nSkipped',
                 ha='center', va='center', fontsize=12,
                 color='gray', transform=ax1.transAxes)
        ax1.axis('off')
    else:
        bar_labels  = ['AE Error\n(MSE)', 'Classifier\nProb']
        bar_values  = [result['ae_error'], result['clf_prob']]
        bar_flags   = [result['ae_flag'],  result['clf_flag']]
        thresholds  = [ae_threshold,       clf_threshold]
        bar_colours = ['crimson' if f else 'steelblue' for f in bar_flags]

        x_pos = range(len(bar_labels))
        bars  = ax1.bar(x_pos, bar_values, color=bar_colours, width=0.5)
        ax1.set_xticks(list(x_pos))
        ax1.set_xticklabels(bar_labels, fontsize=9)

        for i, (thr, bar) in enumerate(zip(thresholds, bars)):
            ax1.plot([bar.get_x(), bar.get_x() + bar.get_width()],
                     [thr, thr], color='orange', linestyle='--', linewidth=1.5,
                     label=f'thr={thr:.4f}' if i == 0 else None)

        bar_max = max(bar_values) if max(bar_values) > 0 else 1
        for bar, val in zip(bars, bar_values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     min(val + bar_max * 0.05, bar_max * 0.93),
                     f'{val:.5f}', ha='center', fontsize=8, fontweight='bold')

        ax1.set_title('Model Scores', fontsize=10)
        ax1.set_ylabel('Score')

    # Panel 3: session stats
    ax2 = fig.add_subplot(gs[2])
    ax2.axis('off')
    stat_text = (
        f"  Session Stats\n"
        f"  {'─' * 20}\n"
        f"  Frames fetched  : {stats['total']}\n"
        f"  Clean        : {stats['clean']}\n"
        f"  Pollution    : {stats['pollution']}\n"
        f"  Dark/skipped : {stats['dark']}\n"
        f"  Errors       : {stats['errors']}\n"
        f"  {'─' * 20}\n"
        f"  Started : {stats['start'].strftime('%H:%M:%S')}\n"
        f"  Last    : {ts.strftime('%H:%M:%S')}\n\n"
        f"  Saved - /{saved_path.parent.name}/\n"
        f"  {saved_path.name}"
    )
    ax2.text(0.05, 0.95, stat_text, transform=ax2.transAxes,
             fontsize=8.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.suptitle('River Bollin ; Live Pollution Monitor', fontsize=13)
    plt.tight_layout()
    plt.show()


def predict_single_frame(img: Image.Image, ae_model, clf_model,
                         ae_threshold: float, clf_threshold: float,
                         device: torch.device) -> dict:
    # Run inference on a single frame using both models
    if is_too_dark(img):
        return {
            'verdict':    'SKIPPED',
            'pollution':  None,
            'ae_error':   None,
            'ae_flag':    None,
            'clf_prob':   None,
            'clf_flag':   None,
            'model_used': None,
            'reason':     'Frame too dark',
        }

    # AE score
    ae_in = AE_TF(img).unsqueeze(0).to(device)
    ae_error = ae_model.reconstruction_error(ae_in).item()
    ae_flag = ae_error > ae_threshold

    # Classifier score
    clf_in = BASE_TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        clf_prob = torch.sigmoid(clf_model(clf_in).squeeze()).item()
    clf_flag = clf_prob >= clf_threshold

    pollution = ae_flag or clf_flag

    return {
        'verdict':    'POLLUTION' if pollution else 'Clean',
        'pollution':  pollution,
        'ae_error':   round(ae_error, 6),
        'ae_flag':    ae_flag,
        'clf_prob':   round(clf_prob, 4),
        'clf_flag':   clf_flag,
        'model_used': 'AE + Classifier',
        'reason':     None,
    }


def run_live_monitor(
    ae_model,
    clf_model,
    webcam_url: str = 'http://www.maccinfo.com/WebcamUtility/Utility.jpg',
    poll_interval: int = 60,
    request_timeout: int = 15,
    ae_threshold: float = 0.001,
    clf_threshold: float = 0.5,
    base_dir: str = '/content/drive/MyDrive/bollin_pollution',
    device: torch.device = DEVICE
):
    """
    Main loop for live webcam monitoring.
    Stops with Runtime - Interrupt execution.
    """
    ae_model.eval()
    clf_model.eval()

    base_path = Path(base_dir)
    clean_dir = base_path / 'images' / 'clean'
    pollution_dir = base_path / 'images' / 'pollution'
    dark_dir = base_path / 'images' / 'dark'
    log_path = base_path / 'monitor_log.csv'
    alert_log = base_path / 'alerts.log'

    for d in [clean_dir, pollution_dir, dark_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': 0,
        'clean': 0,
        'pollution': 0,
        'dark': 0,
        'errors': 0,
        'start': datetime.now(),
    }

    last_hash = None

    print(f'   Monitor started — polling every {poll_interval}s')
    print(f'   Webcam   : {webcam_url}')
    print(f'   Saving to: {base_path}')
    print(f'   Log      : {log_path}')
    print(f'   Stop with ■ or Runtime - Interrupt execution\n')

    while True:
        ts = datetime.now()
        img = fetch_frame(webcam_url, timeout=request_timeout)

        if img is None:
            stats['errors'] += 1
            print(f'  [{ts.strftime("%H:%M:%S")}] Fetch failed — retrying in {poll_interval}s')
            time.sleep(poll_interval)
            continue

        h = image_hash(img)
        if h == last_hash:
            print(f'  [{ts.strftime("%H:%M:%S")}] Duplicate frame — waiting …')
            time.sleep(poll_interval)
            continue
        last_hash = h
        stats['total'] += 1

        result = predict_single_frame(img, ae_model, clf_model,
                                      ae_threshold, clf_threshold, device)

        if result['verdict'] == 'SKIPPED':
            saved_path = save_frame(img, dark_dir, 'dark', ts)
            stats['dark'] += 1
        elif result['pollution']:
            saved_path = save_frame(img, pollution_dir, 'POLLUTION', ts)
            stats['pollution'] += 1
            trigger_alert(ts, result, saved_path, alert_log)
        else:
            saved_path = save_frame(img, clean_dir, 'clean', ts)
            stats['clean'] += 1

        log_result({
            'timestamp': ts.isoformat(),
            'verdict': result['verdict'],
            'pollution': result['pollution'],
            'ae_error': result['ae_error'],
            'ae_flag': result['ae_flag'],
            'clf_prob': result['clf_prob'],
            'clf_flag': result['clf_flag'],
            'model_used': result['model_used'],
            'saved_path': str(saved_path),
        }, log_path)

        display_status(img, result, ts, stats, saved_path, ae_threshold, clf_threshold)
        print(f'  [{ts.strftime("%H:%M:%S")}] {result["verdict"]} | '
              f'ae={result["ae_error"]} | clf={result["clf_prob"]} | - {saved_path.name}')

        time.sleep(poll_interval)
