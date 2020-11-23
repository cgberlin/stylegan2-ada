import cv2
import numpy as np
import os
from os.path import isfile, join
from tqdm import tqdm
from argparse import ArgumentParser
import subprocess


def create_video(fps, resolution, input_path, music_path, out_dir, out_file):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    frame_array = []
    files = [f for f in os.listdir(input_path) if isfile(join(input_path, f))]

    files.sort(key=lambda x: x[5:-4])

    for i in tqdm(range(len(files))):
        filename = input_path + "{}.png".format(i)
        img = cv2.imread(filename)
        img = cv2.resize(img, (resolution, resolution))

        height, width, layers = img.shape
        size = (width, height)

        frame_array.append(img)
    out = cv2.VideoWriter(f'{out_dir}/noaud.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in tqdm(range(len(frame_array))):
        out.write(frame_array[i])
    out.release()

    subprocess.call(
        ['ffmpeg', '-i', f'{out_dir}/noaud.mp4', '-i', music_path, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
         out_file])


# ----------------------------------------------------------------------------


def main():

    parser = ArgumentParser(
        description='Mux video from generated images'
    )
    parser.add_argument('--music_path', help='Wav file to process', required=True)
    parser.add_argument('--out_dir', help='Where to save the output file', default='./output')
    parser.add_argument('--out_file', help='Name for the generated video', default='generated.mp4')
    parser.add_argument('--input_path', help='Path to the input frames', default='./imgs')
    parser.add_argument('--fps', help='Framerate for the video', default=40)
    parser.add_argument('--resolution', help='Resolution to encode', default=512)
    args = parser.parse_args()
    create_video(args.fps,
                 args.resolution,
                 args.input_path,
                 args.music_path,
                 args.out_dir,
                 args.out_file)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
