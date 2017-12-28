from moviepy.editor import VideoFileClip
from object_detection import ObjectDetector
from lane_finder import LaneFinder
from object_detect_yolo import YoloDetector

if __name__ == "__main__":
    def remove_mp4_extension(file_name):
        return file_name.replace(".mp4", "")

    # Read movie
    # yolo = YoloDetector()
    # lane_finder = LaneFinder(save_original_images=True, object_detection_func=yolo.process_image_array)
    object_detection = ObjectDetector()

    video_file = 'project_video.mp4'
    clip = VideoFileClip(video_file, audio=False)
    t_start = 0
    t_end = 1
    if t_end > 0.0:
        clip = clip.subclip(t_start=t_start, t_end=t_end)
    else:
        clip = clip.subclip(t_start=t_start)

    clip = clip.fl_image(object_detection.process_image)
    clip.write_videofile("{}_output.mp4".format(remove_mp4_extension(video_file)), audio=False)








