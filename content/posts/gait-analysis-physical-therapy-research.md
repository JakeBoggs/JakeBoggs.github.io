---
title: "Gait Analysis for Physical Therapy with YOLOv11"
date: "2025-04-21"
draft: false
summary: "Built a video analysis tool for my mom's physical therapy research. Uses YOLOv11-pose for automatic joint detection with a drag-to-correct interface."
---
Analyzing how people walk using video is common in research and clinical settings, but getting accurate joint angles usually means either expensive equipment or annotating frames, which is slow and tedious. After seeing my mother spend a lot of time doing this manually over Easter weekend, I built a Python tool to help her speed it up. It uses [YOLOv11-pose](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-pose-estimation) for automatic detection and adds an interactive interface for manual adjustments.

**Demo**

<video width="100%" controls>
  <source src="/videos/gait-analyzer.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**How it Works**

The script lets you load a video file (specifically a side-view of someone walking) and capture joint angles at various points. Here's the workflow:

1.  **Load & Play:** Open the video. You can play/pause with `Space` and move frame-by-frame using the `Left`/`Right` arrow keys or `p`/`n`.
2.  **Automatic Detection:** When the video is paused or you step to a new frame, the script runs the YOLOv11-pose model on the frame.
3.  **Key Points:** The model identifies the most prominent person and then 5 key points relevant for side-view gait are calculated from the resulting keypoints. For example, calculating the hip center:
    ```python
    # kps holds the keypoints for the detected person
    Lh = kps[KP['left_hip']][:2] # Get x,y for left hip
    Rh = kps[KP['right_hip']][:2] # Get x,y for right hip
    H = ((Lh+Rh)/2).tolist() # Calculate the midpoint
    ```
    And estimating the toe position based on the knee (Kpt) and ankle (Apt):
    ```python
    # Simplified Toe Calculation
    Kpt_np, Apt_np = np.array(Kpt), np.array(Apt)
    leg_vector_ka = Apt_np - Kpt_np
    # Calculate a perpendicular vector based on side
    perp_vector = np.array([-leg_vector_ka[1], leg_vector_ka[0]]) # Example for left side
    # Offset from ankle, half the length of the knee-ankle segment
    T_np = Apt_np + 0.5 * perp_vector
    T = T_np.tolist()
    ```
    These points (S, H, K, A, T) are drawn on the video frame.
4.  **Angle Calculation:** It also computes and displays three angles: Shoulder-Hip-Knee, Hip-Knee-Ankle, and Knee-Ankle-Toe.
    ```python
    # Simplified angle calculation within recompute_angles()
    S, H, K, A, T = [np.array(p) for p in points]
    # Calculate vectors (e.g., from Hip to Shoulder, Hip to Knee)
    vec_HS = S - H
    vec_HK = K - H
    # Use dot product formula (via angle_between helper) for angle
    angle_SHK_internal = angle_between(vec_HS, vec_HK) # Custom helper function
    angles[0] = 180.0 - angle_SHK_internal # Store final angle
    # ... similar calculations for HKA and KAT ...
    ```
5.  **Interactive Correction:** If the automatically detected joints aren't quite right, you can **click and drag** them directly on the video frame.
6.  **Save Data:** Once you're satisfied with the point placement on a given frame (while paused), press the **'s'** key. This triggers the saving process detailed below.

**Output Data**

When you run the script, you can specify an output directory (or it defaults to one named `output`). Inside that directory, the script automatically creates a **timestamped subfolder** for each video analysis session. For example, analyzing `my_gait_video.mp4` might create a folder like:

`output/my_gait_video_20231028_163045/`

This keeps results from different runs or videos organized. Inside this folder, you'll find:

*   **Cropped Images (PNGs):** Every time you press 's', a PNG file is saved, named like `my_gait_video_0123.png` (where `0123` is the frame number). This isn't the whole video frame, but a cropped image focused on the detected person's bounding box. The 5 points (S, H, K, A, T) and the calculated angles are drawn directly onto this image. These are useful for visual checks, qualitative assessments, or including specific examples in reports.
*   **Angle Data (CSV):** A single CSV file, named after the original video (e.g., `my_gait_video.csv`), is created for the session. Each time you press 's', a new row is added to this file. The columns are:
    `Frame, S-H-K, H-K-A, K-A-T`
    So, a row might look like: `123, 165.23, 170.51, 85.90`

**Important Note on Privacy:** The YOLO model needs to be downloaded on first use, but after that, all video processing and analysis happens **locally on your machine**. No video data is sent anywhere.

**How to Use It**

1.  **Get the code:** It's available on GitHub:
    [https://github.com/JakeBoggs/Gait-Analyzer](https://github.com/JakeBoggs/Gait-Analyzer)
2.  **Install dependencies:** Download [Python](https://www.python.org/downloads/) if you do not already have it, then install the required libraries with `pip install opencv-python numpy ultralytics`
2.  **Run from the terminal:** Navigate to the repository directory and run:
    ```bash
    python track.py --video <path_to_your_video.mp4> --output <your_results_folder>
    ```
    Replace `<path_to_your_video.mp4>` with the actual file path and `<your_results_folder>` with where you want the main output directory created. Check the README on GitHub for other options.