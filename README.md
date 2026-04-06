# SAM Video Compare Demo

This project is a simple test interface to compare two videos using SAM 2.

## What it does

- Upload one expert video and one learner video
- Extract the first frame of each video
- Let the user click on the object to segment
- Apply SAM 2 on both first frames
- Show overlay results
- Compare:
  - mask area
  - mask centroid
  - bounding box

## Main files

- `app.py` → Gradio interface
- `sam_image_utils.py` → SAM image segmentation logic
- `video_utils.py` → video helper functions
- `compare_utils.py` → comparison helper functions

## Install

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt