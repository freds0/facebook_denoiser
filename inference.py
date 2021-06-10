import argparse
from denoiser.demucs import DemucsStreamer
from denoiser import pretrained
from denoiser.demucs import Demucs
import torch
import torchaudio
import os
import glob
import time
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/"
MASTER_64_URL = ROOT + "master64-8a5dfb4bb92753dd.th"

def _demucs(pretrained, url, **kwargs):
    model = Demucs(**kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict)
    return model

def get_model(model_path = None):
    """
    Load local model package or torchhub pre-trained model.
    """
    pretrained = True
    if model_path is not None:
        logger.info("Loading model from %s", model_path)
        pkg = torch.load(model_path)
        model = deserialize_model(pkg)
    else:
        print("Loading pre-trained real time H=64 model trained on DNS and Valentini.")
        model = _demucs(pretrained, MASTER_64_URL, hidden=64)
    return model

def get_estimate(model, noisy, streaming = False, dry=0):
    torch.set_num_threads(1)
    if streaming:
        streamer = DemucsStreamer(model, dry=dry)
        with torch.no_grad():
            estimate = torch.cat([
                streamer.feed(noisy[0]),
                streamer.flush()], dim=1)[None]
    else:
        with torch.no_grad():
            estimate = model(noisy)
            estimate = (1 - dry) * estimate + dry * noisy
    return estimate

def enhance(noisy_signal, model=None):
    model_path = None
    # Load model
    if not model:
        model = get_model(model_path).to(device)
    model.eval()

    noisy_signal = noisy_signal.to(device)
    # Forward
    estimate = get_estimate(model, noisy_signal)
    #save_wavs(estimate, noisy_signals, filenames, out_dir, sr=args.sample_rate)
    return estimate

def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)
    
def save_wav(estimate, filepath, sr=16_000):
    # Write result
    #filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
    write(estimate, filepath, sr=sr)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--input_dir', default='input', help='Input dir')
    parser.add_argument('--output_dir', default='output', help='Output dir')    
    args = parser.parse_args()

    model = get_model().to(device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for filepath in tqdm(glob.glob(args.input_dir + '/*.wav')):
        filename = os.path.basename(filepath)
        new_filepath = os.path.join(args.output_dir, filename)
        noisy_signal, sr = torchaudio.load(filepath)
        unoise_signal = enhance(noisy_signal, model)
        save_wav(unoise_signal.to('cpu').squeeze(), new_filepath, sr)




if __name__ == "__main__":
    main()

