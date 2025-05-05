---
title: "Gait Analysis for Physical Therapy with YOLOv11"
date: "2025-04-21"
draft: false
---
Analyzing how people walk using video is common in research and clinical settings, but getting accurate joint angles usually means either expensive equipment or manually annotating frames, which is slow and tedious.

Over Easter weekend, I built a Python tool to help my mother with her research study analyzing patient videos. It uses YOLOv11-pose for automatic detection and adds an interactive interface for manual adjustments.

**What it Does**

The script (`track.py`) lets you load a video file (ideally a side-view of someone walking). Here’s the workflow:

1.  **Load & Play:** Open the video. You can play/pause with `Space` and move frame-by-frame using the `Left`/`Right` arrow keys or `p`/`n`.
2.  **Automatic Detection:** When the video is paused or you step to a new frame, the script runs YOLOv11-pose on the frame using the `ultralytics` library.
    ```python
    # Simplified: Loading the model
    model = YOLO(args.model) # e.g., YOLO("yolov11l-pose.pt")

    # Simplified: Running detection on a frame (or crop)
    results = model(current_frame_or_crop, verbose=False)[0]
    keypoints_data = results.keypoints.data
    ```
3.  **Key Points:** It identifies the main person and calculates 5 key points relevant for side-view gait from the `keypoints_data`. For example, calculating the hip center:
    ```python
    # kps holds the keypoints for the detected person
    Lh = kps[KP['left_hip']][:2] # Get x,y for left hip
    Rh = kps[KP['right_hip']][:2] # Get x,y for right hip
    H = ((Lh+Rh)/2).tolist() # Calculate the midpoint
    ```
    And estimating the toe position based on the knee (Kpt) and ankle (Apt):
    ```python
    # Simplified Toe Calculation (vector math)
    Kpt_np, Apt_np = np.array(Kpt), np.array(Apt)
    leg_vector_ka = Apt_np - Kpt_np
    # Calculate a perpendicular vector based on side
    perp_vector = np.array([-leg_vector_ka[1], leg_vector_ka[0]]) # Example for left side
    # Offset from ankle, half the length of the knee-ankle segment
    T_np = Apt_np + 0.5 * perp_vector
    T = T_np.tolist()
    ```
    These points (S, H, K, A, T) are drawn on the video frame.
4.  **Angle Calculation:** It also calculates and displays three angles using vector math with `numpy`.
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
5.  **Interactive Correction:** This is the key part. If the automatically detected points aren't quite right, you can **click and drag** them directly on the video frame. This uses an OpenCV mouse callback:
    ```python
    # Simplified mouse callback logic
    def mouse_cb(event, x, y, flags, param):
        global points, dragging, drag_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is near a point, set dragging=True, drag_idx=i
            ...
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            points[drag_idx] = (x,y) # Update point position
            recompute_angles()      # Recalculate angles
            draw_main_window()      # Redraw the display
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            ...
    cv2.setMouseCallback(window_name, mouse_cb) # Register the callback
    ```
6.  **Save Data:** When you're satisfied with the point placement on a given frame (while paused), press the **'s'** key. This triggers the saving process detailed below.

**Demo**

<video width="100%" controls>
  <source src="/videos/gait-analyzer.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**Output: Data Ready for Analysis**

When you run the script, you specify an output directory (or it defaults to one named `output`). Inside that directory, the script automatically creates a **timestamped subfolder** for each video analysis session. For example, analyzing `my_gait_video.mp4` might create a folder like:

`output/my_gait_video_20231028_163045/`

This keeps results from different runs or videos organized. Inside this timestamped folder, you'll find:

*   **Cropped Images (PNGs):** Every time you press 's', a PNG file is saved, named like `my_gait_video_0123.png` (where `0123` is the frame number). This isn't the whole video frame, but a cropped image focused on the detected person's bounding box. The 5 points (S, H, K, A, T) and the calculated angles are drawn directly onto this image. These are useful for visual checks, qualitative assessments, or including specific examples in reports.
*   **Angle Data (CSV):** A single CSV file, named after the original video (e.g., `my_gait_video.csv`), is created for the session. Each time you press 's', a new row is added to this file. The columns are:
    `Frame, S-H-K, H-K-A, K-A-T`
    So, a row might look like: `123, 165.23, 170.51, 85.90`

**Using the Output for Research**

The key output for quantitative research is the **CSV file**. Because it's a standard format:

*   **Spreadsheet Friendly:** You can directly open the `.csv` file in software like Microsoft Excel, Google Sheets, LibreOffice Calc, etc. The data will appear in columns (Frame, S-H-K angle, H-K-A angle, K-A-T angle).
*   **Easy Analysis:** Once in a spreadsheet, you can easily sort by frame, calculate average angles, standard deviations, min/max values, or create plots of angles over time.
*   **Statistical Software:** The CSV format is readily imported into statistical packages like R, SPSS, or using Python libraries like Pandas. This allows for more advanced statistical analysis, comparing different subjects or conditions, and generating publication-quality graphs.

The combination of visual confirmation (PNGs) and structured numerical data (CSV) makes it easier to process video data for research purposes and link back to the source for verification.

**Important Note on Privacy:** The YOLO model needs to be downloaded on first use, but after that, all video processing and analysis happens **locally on your machine**. No video data is sent anywhere.

**How to Use It**

1.  **Get the code:** It's available on GitHub:
    [https://github.com/JakeBoggs/Gait-Analyzer](https://github.com/JakeBoggs/Gait-Analyzer)
2.  **Run from terminal:** Assuming you have Python and the necessary libraries set up, navigate to the code directory and run:
    ```bash
    python track.py --video <path_to_your_video.mp4> --output <your_results_folder>
    ```
    Replace `<path_to_your_video.mp4>` with the actual file path and `<your_results_folder>` with where you want the main output directory created. Check the README on GitHub for other options (like choosing the YOLO model).