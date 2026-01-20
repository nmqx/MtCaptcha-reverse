# MtCaptcha Reversal & Solver

A full reversal of MTCaptcha, including the folding algorithm and the captcha solver.
## 🚀 Features

- **Hybrid Architecture**:
  - **Go Service**: Handles the computationally intensive "Folding" algorithm (PoW) required to generate the valid `fa` token.
  - **Python Controller**: Manages the HTTP session, request signing, and flow orchestration.
  - **PyTorch CRNN**: A Convolutional Recurrent Neural Network trained to solve the specific visual style of MTCaptcha images.
- **Full Flow Automation**: Automates the entire process from fetching the challenge (`getchallenge`), downloading the image (`getimage`), calculating the necessary proofs, and submitting the solution (`solvechallenge`).
- **Resilient Requesting**: Uses `curl_cffi` to mimic real browser TLS fingerprints (Safari/Chrome).

## 🛠️ File Structure

- **`flow.py`**: The main entry point. Orchestrates the solution process, interacting with the MTCaptcha API and the local services.
- **`calculate.go`**: A standalone HTTP service (running on port 9091) that implements the reverse-engineered `generateHypothesis3` algorithm to calculate the `fa` parameter.
- **`cnn.py`**: The neural network definition (CRNN) and training logic.
- **`predictor.py`**: A lightweight inference wrapper for the trained model.
- **`crnn_weights.pt`**: Pre-trained weights for the captcha solver.
- **`captcha.gif`**: Temporary storage for the downloaded captcha image.

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

## ⚡ Usage

This solution requires the Go calculation service to be running in the background before executing the Python script.

### 1. Start the Calculation Service
Open a terminal and run the Go server. It will listen on `localhost:9091`.

```bash
go run calculate.go
```
*You should see output indicating the service is listening.*

### 2. Run the Solver
Open a separate terminal and run the main flow script.

```bash
python flow.py
```

### 3. (Optional) Configuration
- **Proxies**: The `flow.py` script mimics a specific proxy configuration. You **must** update the `proxies` dictionary in `flow.py` with your own working proxies to avoid connection errors or IP bans.
- **Model Training**: If you wish to retrain the model, place your labeled dataset in a folder named `captured_hits` and run `python cnn.py`.

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is intended to demonstrate vulnerabilities in captcha systems and the importance of bot protection. The authors are not responsible for any misuse of this code.