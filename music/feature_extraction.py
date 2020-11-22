"""
https://neurips2019creativity.github.io/doc/Stylizing%20Audio%20Reactive%20Visuals.pdf
"""
from argparse import ArgumentParser
import os
import librosa
import numpy as np
import soundfile as sf
from scipy import signal


def low_pass_filter(y, sr, frame_length, out_dir):
    print("=== Processing low pass filter ===")
    fc = 150
    w = fc / (sr / 2)
    b, a = signal.butter(5, w, 'low')
    lpf_y = signal.filtfilt(b, a, y)

    sf.write(f'{out_dir}/lpf_y.wav', lpf_y, sr)
    lpf_y, sr = librosa.load(f'{out_dir}/lpf_y.wav')

    # lpf_slice_track = librosa.util.frame(lpf_y, frame_length=int(frame_length), hop_length=int(frame_length))
    lpf_y_rmse = librosa.feature.rms(lpf_y, frame_length=int(frame_length), hop_length=int(frame_length), center=True)
    lpf_y_rmse = lpf_y_rmse[0, :]

    np.save(f'{out_dir}/lpf_y_rmse.npy', lpf_y_rmse)


def band_pass_filter(y, sr, frame_length, out_dir):
    print("=== Processing band pass filter ===")
    fc = 350
    w = fc / (sr / 2)
    b, a = signal.butter(5, w, 'low')
    bpf_y = signal.filtfilt(b, a, y)

    fc = 200
    w = fc / (sr / 2)
    b, a = signal.butter(5, w, 'high')
    bpf_y = signal.filtfilt(b, a, bpf_y)

    sf.write(f'{out_dir}/bpf_y.wav', bpf_y, sr)
    bpf_y, sr = librosa.load(f'{out_dir}/bpf_y.wav')

    # bpf_slice_track = librosa.util.frame(bpf_y, frame_length=int(frame_length), hop_length=int(frame_length))
    bpf_y_rmse = librosa.feature.rms(bpf_y, frame_length=int(frame_length), hop_length=int(frame_length), center=True)
    bpf_y_rmse = bpf_y_rmse[0, :]

    np.save(f'{out_dir}/bpf_y_rmse.npy', bpf_y_rmse)


def high_pass_filter(y, sr, frame_length, out_dir):
    print("=== Processing high pass filter ===")
    fc = 5000
    w = fc / (sr / 2)
    b, a = signal.butter(5, w, 'low')
    hpf_y = signal.filtfilt(b, a, y)

    fc = 500
    w = fc / (22050 / 2)
    b, a = signal.butter(5, w, 'high')
    hpf_y = signal.filtfilt(b, a, hpf_y)

    sf.write(f'{out_dir}/hpf_y.wav', hpf_y, sr)
    hpf_y, sr = librosa.load(f'{out_dir}/hpf_y.wav')

    hpf_slice_track = librosa.util.frame(hpf_y, frame_length=int(frame_length), hop_length=int(frame_length))
    hpf_y_rmse = librosa.feature.rms(hpf_y, frame_length=int(frame_length), hop_length=int(frame_length), center=True)
    hpf_y_rmse = hpf_y_rmse[0, :]

    np.save(f'{out_dir}/hpf_y_rmse.npy', hpf_y_rmse)


def extract_features(music_path, out_dir='./features'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    y, sr = librosa.load(music_path)
    # tempo, beat_frames = beat_track(y, sr)
    bpm = 150  # int(tempo)
    print(f'setting bpm {str(bpm)}')
    set_frame_track = 1 / (bpm * 16 / 60)
    frame_length = sr * set_frame_track
    low_pass_filter(y, sr, frame_length, out_dir)
    band_pass_filter(y, sr, frame_length, out_dir)
    high_pass_filter(y, sr, frame_length, out_dir)

# ----------------------------------------------------------------------------


def main():

    parser = ArgumentParser(
        description='Extract audio features from wav files'
    )
    parser.add_argument('--music_path', help='Wav file to process', required=True)
    parser.add_argument('--out_dir', help='Where to save the output features', default='./features')
    args = parser.parse_args()
    extract_features(args.get('music_path'), args.get('out_dir'))


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
