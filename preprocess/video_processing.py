import cv2
import os
from tqdm import tqdm


class VideoToImages:
    """use first frame of the videos and store them as images (use only first frame because most frames are very similar)"""
    def __init__(self, video_path, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.video_path = video_path
        self.output_folder = output_folder
    
    def video_to_image(self, video_file_name, img_name):
        assert img_name.endswith(".png"), "Image name must end with .png!"
        
        video_file = f"{self.video_path}/{video_file_name}"
        vidcap = cv2.VideoCapture(video_file)   # open the video file
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))        # get the total number of frames
        success, image = vidcap.read()          # loop through the video and save each frame as an image
        if not success:
            raise Exception("Failed to read video file!")
        image_filename = os.path.join(self.output_folder, img_name)          # construct the output image filename

        cv2.imwrite(image_filename, image)          # save
        vidcap.release()
        
    def convert_all_to_images(self):
        c = 1
        for video_file_name in tqdm(os.listdir(self.video_path), desc="Converting videos to images..."):
            img_name = f"{c}.png"
            try:
                self.video_to_image(video_file_name, img_name)
                c += 1
            except:
                pass