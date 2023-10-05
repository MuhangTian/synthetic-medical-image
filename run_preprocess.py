from preprocess.video_processing import VideoToImages
from preprocess.train_test import train_test_split

if __name__ == "__main__":
    train_test_split("data/imgs")
    # converter = VideoToImages("data/videos", "data/imgs")
    # converter.convert_all_to_images()
    