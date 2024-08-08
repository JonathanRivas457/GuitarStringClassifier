import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def get_spectrogram(audio_path, save_dir):

    # Create file name for saving to directory
    file_name = audio_path.split('/')[1].strip()
    file_name = file_name.split('.')[0].strip()
    file_name += '.png'
    y, sr = librosa.load(audio_path)

    # Create spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Create time frame to filter out beginning and end of audio
    start_time = 0.5
    end_time = 2
    start_frame = librosa.time_to_frames(start_time, sr=sr)
    end_frame = librosa.time_to_frames(end_time, sr=sr)

    # Save spectrogram to folder
    plt.figure(figsize=(5, 2))
    librosa.display.specshow(spectrogram_db[:, start_frame:end_frame], sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('mel spectrogram')
    plt.savefig(save_dir + file_name)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='extract spectograms from audio files')
    parser.add_argument("wav_directory", help='directory containing audio files')
    parser.add_argument("save_directory", help='directory to save images to')
    args = parser.parse_args()
    wav_directory = args.wav_directory + '/'
    save_directory = args.save_directory + '/'

    for file in os.listdir(wav_directory):
        file_path = wav_directory + file
        print(file_path)
        get_spectrogram(file_path, save_directory)


if __name__ == "__main__":
    main()
