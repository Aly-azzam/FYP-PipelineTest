# Fast-HaMeR Windows Setup Guide

This guide is for running the Fast-HaMeR repo on **Windows 11** with an **NVIDIA GPU**.

It is based on the working setup used in this project.

## What this setup assumes

- Windows 11
- NVIDIA GPU
- Python 3.10
- Conda installed
- Visual Studio 2019 Build Tools installed
- CUDA Toolkit **12.1** installed with **nvcc**

## Important notes before starting

This project is **not** a simple `pip install -r requirements.txt` setup.

There are some manual steps that matter:

- CUDA Toolkit 12.1 must be installed
- PyTorch3D must be built manually on Windows
- NumPy must stay **below 2**
- MANO files must be placed manually
- demo data must be extracted manually
- On Windows, use `PYOPENGL_PLATFORM=win32`

---

## 1. Create the conda environment

From a normal terminal:

```powershell
conda env create -f environment.yml
conda activate fasthamer
```

If that fails or you prefer manual installation, do:

```powershell
conda create -n fasthamer python=3.10 -y
conda activate fasthamer
pip install -r requirements-pip.txt
```

---

## 2. Install PyTorch with CUDA 12.1

Use the official PyTorch wheel index:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Expected:

- Torch 2.1.0
- CUDA 12.1
- `True` for GPU availability

---

## 3. Install MMCV

Do **not** let pip compile MMCV from source on Windows.

Install the working wheel directly:

```powershell
pip install https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv_full-1.7.2-cp310-cp310-win_amd64.whl
```

Verify:

```powershell
python -c "import mmcv; print(mmcv.__version__)"
```

Expected:

- `1.7.2`

---

## 4. Install Fast-HaMeR in editable mode

From the Fast-HaMeR repo root:

```powershell
pip install -e . --no-deps
```

`--no-deps` is used intentionally to avoid breaking the working package versions.

---

## 5. Fix OpenCV / NumPy rules

These rules matter:

- Keep **NumPy < 2**
- Use **opencv-python==4.10.0.84**
- Do **not** leave `opencv-contrib-python 4.13.x` installed because it pulls NumPy 2.x and breaks PyTorch/PyTorch3D

Safe commands:

```powershell
pip install "numpy<2"
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
pip install opencv-python==4.10.0.84
```

Verify:

```powershell
python -c "import cv2, numpy; print(cv2.__version__); print(hasattr(cv2, 'dnn')); print(numpy.__version__)"
```

Expected:

- OpenCV 4.10.0
- `True` for `cv2.dnn`
- NumPy 1.26.x

---

## 6. Install remaining Python packages

```powershell
pip install webdataset
pip install rtmlib
```

If installing `rtmlib` upgrades NumPy to 2.x again, fix it immediately:

```powershell
pip install "numpy<2"
pip uninstall -y opencv-contrib-python
pip install opencv-python==4.10.0.84
```

---

## 7. Download and place demo data

Download:

- `hamer_demo_data.tar.gz`

Put it inside the Fast-HaMeR repo root, for example:

```text
C:\Users\YOUR_NAME\Desktop\FYP-PipelineTest\backend\Fast-HaMeR\hamer_demo_data.tar.gz
```

Extract it there:

```powershell
tar -xzf .\hamer_demo_data.tar.gz
```

After extraction, `_DATA` should contain folders such as:

- `_DATA\data`
- `_DATA\hamer_ckpts`
- `_DATA\vitpose_ckpts`

---

## 8. Place MANO files

Create the MANO folder if needed:

```powershell
mkdir .\_DATA\data\mano -Force
```

Put at least this file here:

```text
_DATA\data\mano\MANO_RIGHT.pkl
```

If you also have `MANO_LEFT.pkl`, keep it there too.

---

## 9. Install CUDA Toolkit 12.1 fully

On Windows, make sure CUDA Toolkit **12.1** is installed with the **compiler (`nvcc`)**.

After installation, in a new terminal:

```cmd
where nvcc
nvcc --version
```

You want to see:

- path to `CUDA\v12.1\bin\nvcc.exe`
- CUDA release 12.1

If another CUDA version appears first, you must force 12.1 in the build terminal.

---

## 10. Build PyTorch3D on Windows

Clone PyTorch3D somewhere outside the Fast-HaMeR repo:

```powershell
cd C:\Users\YOUR_NAME\Desktop
git clone https://github.com/facebookresearch/pytorch3d.git
```

Then open:

**x64 Native Tools Command Prompt for VS 2019**

In that terminal, run:

```cmd
conda activate fasthamer
cd C:\Users\YOUR_NAME\Desktop\pytorch3d

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\libnvvp;%PATH%
set DISTUTILS_USE_SDK=1

python -m pip install ninja
python -m pip install iopath fvcore
rmdir /s /q build
python setup.py install
```

Verify:

```cmd
python -c "import pytorch3d; print('pytorch3d ok')"
```

---

## 11. Verify core imports inside Fast-HaMeR

Back in the Fast-HaMeR repo:

```powershell
python -c "import torch, mmcv, cv2, numpy, hamer, pytorch3d; print('core imports ok')"
```

---

## 12. Run the first demo

Use PowerShell and force the OpenGL backend for Windows:

```powershell
$env:PYOPENGL_PLATFORM="win32"
python .\demo_image.py --img_folder .\example_data --out_folder .\demo_out
```

Then inspect outputs:

```powershell
dir .\demo_out
```

Note: in this repo version, `demo_image.py` uses **`--img_folder`**, not `--img`.

---

## 13. Common problems and fixes

### Problem: `No module named pytorch3d`

Fix: PyTorch3D was not built successfully. Repeat Step 10 in the Visual Studio terminal.

### Problem: `No module named rtmlib`

Fix:

```powershell
pip install rtmlib
```

Then re-check NumPy and OpenCV.

### Problem: `No module named webdataset`

Fix:

```powershell
pip install webdataset
```

### Problem: `module 'cv2' has no attribute 'dnn'`

Fix OpenCV completely:

```powershell
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
pip cache purge
pip install opencv-python==4.10.0.84
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'dnn'))"
```

### Problem: NumPy upgraded to 2.x

Fix immediately:

```powershell
pip install "numpy<2"
pip uninstall -y opencv-contrib-python
pip install opencv-python==4.10.0.84
```

### Problem: ONNXRuntime warns that `CUDAExecutionProvider` is unavailable

That means `rtmlib` is running on CPU. The demo can still work. This is not a fatal error.

### Problem: `EGL` / OpenGL errors on Windows

Run the demo with:

```powershell
$env:PYOPENGL_PLATFORM="win32"
```

before starting Python.

---

## 14. Exact working demo command

```powershell
$env:PYOPENGL_PLATFORM="win32"
python .\demo_image.py --img_folder .\example_data --out_folder .\demo_out
```

---

## 15. Final reminder

This setup is fragile if package versions drift.

If something suddenly breaks, first check:

- `numpy` stayed below 2
- `opencv-python` stayed at 4.10.0.84
- `mmcv` still imports
- `pytorch3d` still imports
- CUDA 12.1 is still the active toolkit for the build terminal
