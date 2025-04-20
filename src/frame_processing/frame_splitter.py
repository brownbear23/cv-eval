import os, cv2


def extract_frames(video_path, output_path, frame_interval=1.0):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_frame_folder = os.path.join(output_path, video_name)

    os.makedirs(video_frame_folder, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"    Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  # Duration of video in seconds
    print(f"    Video Duration: {duration:.2f} seconds, FPS: {fps:.2f}, Total Frames: {total_frames}")

    # Calculate frame interval in terms of frame numbers
    frame_interval_frames = int(frame_interval * fps)

    frame_count = 0
    saved_frames = 0

    while True:
        ret, frame = video.read()

        if not ret:  # End of video
            break

        # Save the frame if it corresponds to the interval
        if frame_count % frame_interval_frames == 0:
            frame_filename = os.path.join(str(video_frame_folder), f"{video_name}_{frame_count:03d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1
        frame_count += 1
    video.release()
    print(f"    Extraction complete. Total frames saved: {saved_frames}")


def slice_video_to_frames(in_video_directory="../../media/videos", out_frame_directory="../../media/frames",
                          frame_interval_sec=.25):
    if os.path.isdir(out_frame_directory):
        print(f"WARNING: \"{out_frame_directory}\" directory exists\nDelete the directory to run\nExiting...")
        return False

    count = 1
    for filename in os.listdir(in_video_directory):
        video_path = os.path.join(in_video_directory, filename)
        if os.path.isfile(video_path):
            print(count, "-", filename)
            extract_frames(video_path, out_frame_directory, frame_interval=frame_interval_sec)  # Split every second
            count += 1

    return True

# slice_video_to_frames()
