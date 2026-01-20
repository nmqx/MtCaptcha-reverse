# MtCaptcha Reversal & Solver

A full reversal of MTCaptcha, including the folding algorithm and the visual captcha solver (ocr).


Coded By Nmqx and Thibhrb in python and golang

##  Features

- **Architecture** :
  - **Go Service**: Handles the computationally intensive "Folding" algorithm (PoW) required to generate the valid `fa` token.
  - **Python Controller**: Manages the HTTP session, request signing, and flow orchestration.
  - **PyTorch CRNN**: A Convolutional Recurrent Neural Network trained to solve the specific visual style of MTCaptcha images.
- **Full Flow Automation**: Automates the entire process from fetching the challenge (`getchallenge`), downloading the image (`getimage`), calculating the necessary proofs, and submitting the solution (`solvechallenge`).
- **Network**: Uses `curl_cffi` to mimic real browser TLS fingerprints (Safari/Chrome).

## 🛠️ File Structure

- **`flow.py`**: The main entry point. Orchestrates the solution process, interacting with the MTCaptcha API and the local services.
- **`calculate.go`**: A standalone HTTP service (running on port 9091) that implements the reverse-engineered `generateHypothesis3` algorithm to calculate the `fa` parameter.
- **`cnn.py`**: The neural network definition (CRNN) and training logic.
- **`predictor.py`**: A lightweight inference wrapper for the trained model.
- **`crnn_weights.pt`**: Pre-trained weights for the captcha solver.


## 📋 Prerequisites

- **Python 3.8+**
- **Go 1.18+**
- **NVIDIA GPU** (Optional, but recommended for faster CRNN inference)

## 📦 Installation

1. **Install Python Dependencies**:
   ```bash
   pip install torch torchvision torchaudio numpy pillow requests curl_cffi
   ```

2. **Initialize Go Module**:
   ```bash
   go mod init mtcaptcha-solver
   go mod tidy
   ```

## Usage
### 1. Run the server
The js solving runs or a localhost port , run the go script
```bash
go run calculate.go
```
and it will listen on port 9091

### 2. Run the Solver
Open a separate terminal and run the main flow script.

```bash
python flow.py
```

### 3. (Optional) Configuration
- **Proxies**: Change in flow.py, use session proxies and use the randint var to make a session id so it stays the same for the session.
- **Model Training**: If you wish to retrain the model, place your labeled dataset in a folder named `captured_hits` and run `python cnn.py`.
- **Dataset**: Used a dataset by scraping the captchas so lowkey do your own dataset if you want to improve the model.

## ⚠️ Big Big Disclaimer

This project is for **educational and research purposes only**. It is intended to demonstrate vulnerabilities in captcha systems and the importance of bot protection. The authors are not responsible for any misuse of this code.

## ⚠️ NB
The proxy used in the script has like 10mb left so dont bother, use your own proxies. Also the cnn model is not the best, it's just a proof of concept, it will have a lot of false positives and false negatives. (around 10%), last thing, a fingerprint banned is a  1432 code on the getimage request. I have no clue why but probably because curl_cffi is not a real browser and doesn't emulate perfectly everything idk feel free to try to fix it on a fork