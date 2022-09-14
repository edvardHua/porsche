import sys
import argparse

from porsche.media_processing import concat_videos, resize_video
from pprint import pprint

if __name__ == "__main__":
    par = argparse.ArgumentParser(description="Utility for video processing.")
    par.add_argument("--func", "-f", type=str, default="vidconcat")
    par.add_argument("video_paths", metavar="args", nargs="+", type=str)

    args = par.parse_args()

    if args.func == "vidconcat":
        concat_videos(str(args.video_paths[0]),
                      str(args.video_paths[1]),
                      str(args.video_paths[2]))

    if args.func == "vidresize":
        resize_video(str(args.video_paths[0]),
                     eval(args.video_paths[1]),
                     eval(args.video_paths[2]))
