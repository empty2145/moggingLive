# 🗿 Live Mogging Detector (`mog-cv`)

A real-time computer vision pipeline built with **OpenCV**, **MediaPipe Face Mesh**, **NumPy**, and **Pygame** that evaluates facial geometry on a live webcam feed. 

Depending on your facial posture, the engine dynamically switches between two states:
* **Chud Mode (Default / Bloated):** Overlays Wojak assets, renders warning banners, and loops goofy audio cues when your face is relaxed.
* **Lethal Mogging Mode (Hollow Cheeks / Jaw Clench):** Triggers an automated high-contrast phonk edit—complete with CLAHE shadow carving, unsharp masking, radial vignetting, glowing hunter eyes, and background audio sync.

---

## ⚙️ How It Works

Instead of training a custom classifier from scratch, `mog-cv` uses deterministic vector geometry on MediaPipe's **478 3D facial landmarks**:

1. **Zygomatic-to-Labial Ratio:**
   * Extracts pixel coordinates for the outer cheekbones (**Landmarks `234` & `454`**) and mouth corners (**Landmarks `61` & `291`**).
   * Computes the Euclidean ($L_2$) norm between both pairs:
     $$\text{Ratio} = \frac{\Vert{}\mathbf{p}_{291} - \mathbf{p}_{61}\Vert{}_2}{\Vert{}\mathbf{p}_{454} - \mathbf{p}_{234}\Vert{}_2}$$
   * Sucking in the cheeks (buccal hollowing) draws the oral commissures inward relative to the zygomatic arch, dropping the ratio below the activation threshold.

2. **Real-Time Shader Pipeline:**
   * **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Amplifies local luminance gradients along the mandibular line and hollow cheek regions.
   * **Unsharp Masking:** Subtracts a Gaussian-blurred matrix from the weighted grayscale frame to create sharp, gritty structural edges.
   * **Hunter Eye Glow:** Isolates iris center landmarks (**`468`** and **`473`**), renders high-intensity emitters on a zeroed matrix, applies a $25 \times 25$ Gaussian kernel, and blends via additive weighting.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.9+** installed along with a working webcam.

### 2. Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/empty2145/moggingLive](https://github.com/empty2145/moggingLive)
cd moggingLive
pip install opencv-python mediapipe numpy pygame