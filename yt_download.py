import re
from pytube import YouTube
import os
from shutil import copyfile
import pandas as pd
from tqdm import tqdm

FOLDER = os.path.join("data", "videos")


def download_video(name, video_id, start_time, duration_time):
    """
    start and end have to be in the format mm:ss
    """
    file_path = os.path.join(FOLDER, name)
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    video = (
        YouTube(f"https://www.youtube.com/watch?v={video_id}")
        .streams.filter(file_extension="mp4")
        .first()
    )
    file_name = re.sub(r"[.;:,?!]", "", video.title) + ".mp4"
    if not os.path.exists(os.path.join(FOLDER, file_name)):
        video.download(FOLDER)

    output_file = os.path.join(file_path, name + "-" + video_id + ".mp4")
    if start_time != start_time and duration_time != duration_time:
        copyfile(src=os.path.join(FOLDER, file_name), dst=output_file)
    else:
        original_video = os.path.join(FOLDER, file_name)
        os.system(
            f'ffmpeg -hide_banner -loglevel error -ss {start_time} -i "{original_video}" -to {duration_time} -c copy "{output_file}"'
        )


print("Downloading videos of signs from YouTube")

# Create the dataset based on yt_links.csv
df_links = pd.read_csv("yt_links.csv")
for idx, row in tqdm(df_links.iterrows(), total=df_links.shape[0]):
    download_video(*row)

# Delete the videos used to create the clips for the dataset
for file in os.listdir(FOLDER):
    if file.endswith(".mp4"):
        os.remove(os.path.join(FOLDER, file))
