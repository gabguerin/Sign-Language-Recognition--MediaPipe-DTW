import pytube
import os

playlist = pytube.Playlist(
    "https://www.youtube.com/playlist?list=PL2xCZCGucDagmdxQaDsRqrJef_liIJiDK"
)

for video in playlist.videos:
    video.streams.filter(file_extension="mp4").first().download(os.path.join("data", "videos"))
