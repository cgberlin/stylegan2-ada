import os
import pickle
from argparse import ArgumentParser

import PIL.Image
import numpy as np
from tqdm import tqdm

import dnnlib.tflib as tflib


def get_rmse_feature(feature_path):
    return np.load(feature_path)


def load_model(model_path):
    with open(model_path, 'rb') as pickle_file:
        _G, _D, Gs = pickle.load(pickle_file)
    Gs.print_layers()
    return Gs


def get_randn(shape, seed=None):
    if seed:
        rnd = np.random.RandomState(seed)
        return rnd.randn(*shape)
    else:
        return np.random.randn(*shape)


def get_length(vec):
    return np.linalg.norm(vec)


def normalize_vec(vec, unit_length, factor_length):
    orig_length = get_length(vec)
    return vec * (unit_length / orig_length) * factor_length


def get_noise_vars(Gs):
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    noise_pairs = list(zip(noise_vars, tflib.run(noise_vars)))
    return noise_vars


def move_vec(vec, unit_length, factor_length):
    direction_vec = get_randn(vec.shape)
    direction_vec = normalize_vec(direction_vec, unit_length, factor_length)
    return vec + direction_vec


def init_latents(Gs):
    latents = get_randn((1, Gs.input_shape[1]))
    noise_vars = get_noise_vars(Gs)
    noise_vectors = []
    for i in range(len(noise_vars)):
        noise_vectors.append(get_randn(noise_vars[i].shape))
    return latents, noise_vectors


def gen_image(Gs, latents, noise_vectors, noise_vars, fmt, save_path="example.png"):
    for i in range(len(noise_vars)):
        tflib.set_vars({noise_vars[i]: noise_vectors[i]})
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)

    PIL.Image.fromarray(images[0], 'RGB').save(save_path)


def gen_image_mix(Gs, latents_high, latents_band, latents_low, noise_vectors, noise_vars, fmt, lpf, hpf, bpf,
                  save_path="example.png"):
    for i in range(len(noise_vars)):
        tflib.set_vars({noise_vars[i]: noise_vectors[i]})

    high_latents = np.stack(latents_high)
    band_latents = np.stack(latents_band)
    low_latents = np.stack(latents_low)
    high_dlatents = Gs.components.mapping.run(high_latents, None)
    band_dlatents = Gs.components.mapping.run(band_latents, None)
    low_dlatents = Gs.components.mapping.run(low_latents, None)

    combined_dlatents = low_dlatents

    for style in lpf:
        combined_dlatents[:, style] = low_dlatents[:, style]

    for style in hpf:
        combined_dlatents[:, style] = high_dlatents[:, style]

    for style in bpf:
        combined_dlatents[:, style] = band_dlatents[:, style]

    # combined_dlatents = low_dlatents
    # combined_dlatents[:, 3:6] = high_dlatents[:, 3:6]
    # combined_dlatents[:, 6:] = band_dlatents[:, 6:]

    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
                            minibatch_size=8)
    images = Gs.components.synthesis.run(combined_dlatents, randomize_noise=False, **synthesis_kwargs)
    PIL.Image.fromarray(images[0], 'RGB').save(save_path)


def generate_from_music(model_dir,
                        feature_dir,
                        length_high=5,
                        length_mid=3,
                        length_low=16,
                        hpf=None,
                        bpf=None,
                        lpf=None,
                        out_dir='./imgs'):
    if lpf is None:
        lpf = [0, 1, 2]
    if bpf is None:
        bpf = [3, 4, 5]
    if hpf is None:
        hpf = [6, 7, 8, 9, 10, 11, 12, 13]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    tflib.init_tf()
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    unit_length = 15

    # read our saved features
    high_pass_features = get_rmse_feature(f"{feature_dir}/hpf_y_rmse.npy")
    band_pass_features = get_rmse_feature(f"{feature_dir}/bpf_y_rmse.npy")
    low_pass_features = get_rmse_feature(f"{feature_dir}/lpf_y_rmse.npy")

    diff_high_pass = (high_pass_features[1:] - high_pass_features[:-1])
    diff_high_pass = diff_high_pass / np.amax(diff_high_pass, axis=0)

    diff_band_pass = (band_pass_features[1:] - band_pass_features[:-1])
    diff_band_pass = diff_band_pass / np.amax(diff_band_pass, axis=0)

    diff_low_pass = (low_pass_features[1:] - low_pass_features[:-1])
    diff_low_pass = diff_low_pass / np.amax(diff_low_pass, axis=0)

    # load pretrained model
    Gs = load_model(model_dir)

    # initialize latents and noise
    noise_vars = get_noise_vars(Gs)
    latents, noise_vectors = init_latents(Gs)

    latents_high = latents
    latents_band = latents
    latents_low = latents

    # generate first frame
    gen_image_mix(Gs, latents_high, latents_band, latents_low, noise_vectors, noise_vars, fmt, lpf, hpf, bpf,
                  save_path="./{}/{}.png".format(out_dir, 0))

    # loop through low pass and generate the rest, shifting latent based on saved frequencies
    for i in tqdm(range(diff_low_pass.shape[0])):
        latents_high = move_vec(latents_high, length_high, diff_high_pass[i])
        latents_band = move_vec(latents_band, length_mid, diff_band_pass[i])
        latents_low = move_vec(latents_low, length_low, diff_low_pass[i])

        gen_image_mix(Gs, latents_high, latents_band, latents_low, noise_vectors, noise_vars, fmt, lpf, hpf, bpf,
                      save_path="./{}/{}.png".format(out_dir, i))


def main():

    parser = ArgumentParser(
        description='Create image frames from saved features'
    )
    parser.add_argument('--model_dir', help='Path to pretrained model', required=True)
    parser.add_argument('--feature_dir', help='Path to extracted features', default='./features')
    parser.add_argument('--out_dir', help='Where to save the generated frames', default='./imgs')
    parser.add_argument('--length_high', help='How long to "hold" the high influence', default=5)
    parser.add_argument('--length_mid', help='How long to "hold" the mid influence', default=3)
    parser.add_argument('--length_low', help='How long to "hold" the low influence', default=16)
    parser.add_argument('--hpf', help='Which layers to use for high frequencies', default=[6, 7, 8, 9, 10, 11, 12, 13])
    parser.add_argument('--bpf', help='Which layers to use for mid/band frequencies', default=[3, 4, 5])
    parser.add_argument('--lpf', help='Which layers to use for low frequencies', default=[0, 1, 2])
    args = parser.parse_args()
    generate_from_music(args.model_dir,
                        args.feature_dir,
                        args.length_high,
                        args.length_mid,
                        args.length_low,
                        args.hpf,
                        args.bpf,
                        args.lpf,
                        args.out_dir)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

