import random

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
from whispers import speech_to_txt
import time
from pydub.utils import mediainfo


def print_mp3_length(dir_path):
    durations = {500: 0, 400: 0, 700: 0, 'other': 0}
    for file_path in os.listdir(dir_path):

        audio = AudioSegment.from_mp3(f'{dir_path}/{file_path}')
        duration_in_ms = len(audio)
        if duration_in_ms in durations:
            durations[duration_in_ms] += 1
        else:
            durations['other'] += 1

        print(f"The length of {file_path} is {duration_in_ms} ms.")
    print(durations)


def crop_mp3(input_file_path, start_time_ms, end_time_ms, output_file_path):
    audio = AudioSegment.from_mp3(input_file_path)
    cropped_audio = audio[start_time_ms:end_time_ms]
    cropped_audio.export(output_file_path, format="mp3")


def auto_crop_mp3(input_file_path, output_file_path, chunk=0, silence_thresh=-50, duration=300):
    audio = AudioSegment.from_mp3(input_file_path)
    min_silence_len = 100
    seek_step = 1

    nonsilent_chunks = detect_nonsilent(audio, min_silence_len=min_silence_len,
                                        silence_thresh=silence_thresh, seek_step=seek_step)
    if chunk >= len(nonsilent_chunks):
        return False

    start_time = nonsilent_chunks[chunk][0]
    end_time = start_time + duration
    cropped_audio = audio[start_time:end_time]
    cropped_audio.export(output_file_path, format="mp3")
    return True


def auto_crop_directory(input_directory, output_directory):
    temp_results_dir = '../data/temp'
    silence_threshold = -55
    max_chunks = 4
    matches = 0
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    duration = 500
    for chunk_nb in range(max_chunks):
        print(f'chunk: {chunk_nb}')
        for i in range(4):
            threshold = silence_threshold + i * 10
            print(f'threshold: {threshold}')
            for file in os.listdir(input_directory):
                input_path = f"{input_directory}/{file}"
                output_path = f"{temp_results_dir}/{file}"
                result = auto_crop_mp3(input_path, output_path,
                    chunk=chunk_nb,
                    silence_thresh=threshold,
                    duration=duration)
                if result:
                    txt = speech_to_txt(output_path)
                    if txt == 'eryk':
                        matches += 1
                        print(f"match {matches}: {output_path}")
                        os.rename(output_path, f'{output_directory}/{file}')
                        os.remove(input_path)

            for file in os.listdir(temp_results_dir):
                os.remove(f'{temp_results_dir}/{file}')


def crop_directory(input_directory, output_directory, duration, chunk=0):
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    i = 0
    n = len(os.listdir(input_directory))
    for file in os.listdir(input_directory):
        i += 1
        print(f'Cropping {file} {i}/{n}')
        newname = file.replace('.mp3', f'_{chunk}')
        newname = newname + '.mp3'
        auto_crop_mp3(f'{input_directory}/{file}',
            f'{output_directory}/{newname}',
        duration=duration, chunk=chunk)


if __name__ == '__main__':
    # auto_crop_directory('../data/eryk', '../data/eryk_cropped')
    for i in range(4):
        crop_directory('../data/unmatched', '../data/unmatched_cropped',
            duration=500, chunk=i)

    # lengths = {400: 33, 500: 109, 'other': 69}
    # directory = '../data/unmatched'
    #
    # for file in os.listdir(directory):
    #     if 400 in lengths:
    #         duration = 400
    #         lengths[400] -= 1
    #         if lengths[400] <= 0:
    #             del lengths[400]
    #     elif 500 in lengths:
    #         duration = 500
    #         lengths[500] -= 1
    #         if lengths[500] <= 0:
    #             del lengths[500]
    #     elif 'other' in lengths:
    #         duration = random.randint(430, 500)
    #         lengths['other'] -= 1
    #         if lengths['other'] <= 0:
    #             del lengths['other']
    #     else:
    #         print('break.')
    #         break
    #     auto_crop_mp3(f'{directory}/{file}', f'../data/eryk_training/noise/{file}',
    #     duration=duration
    # )

    # crop_directory('../data/eryk', f'../data/temp', duration=)
    # print_mp3_length('../data/eryk_training/eryk')