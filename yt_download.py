import pytube
import os

playlist = pytube.Playlist(
    "https://www.youtube.com/playlist?list=PL2xCZCGucDagmdxQaDsRqrJef_liIJiDK"
)

video_folder = os.path.join("data", "videos")

# Download a playlist of French Signs
for video in playlist.videos:
    video.streams.filter(file_extension="mp4").first().download(video_folder)

# Rename the files to have videos w/ the name "<sign_name>.mp4"
for video in os.listdir(video_folder):
    os.rename(video, video.replace(" - LSF", ""))
