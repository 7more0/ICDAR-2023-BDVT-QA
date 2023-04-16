import whisper
from os import listdir
from tqdm import tqdm
FILEPATH = "dataset/train/train_videos_part1"
RESPATH = "dataset/train/videos_result"

def get_all_text(file_path, model_type):
    # model_type ["base","small","medium","large","large-v2"]
    model = whisper.load_model(model_type)
    
    file_name_list=listdir(path=file_path)
    for file_name in tqdm(file_name_list):
        if file_name.endswith('.mp4'):
            try:
                # load audio and pad/trim it to fit 30 seconds
                audio = whisper.load_audio(file_path + f"/{file_name}")
                audio = whisper.pad_or_trim(audio)

                # make log-Mel spectrogram and move to the same device as the model
                mel = whisper.log_mel_spectrogram(audio).to(model.device)

                # detect the spoken language
                _, probs = model.detect_language(mel)
                # print(f"Detected language: {max(probs, key=probs.get)}")

                # decode the audio
                options = whisper.DecodingOptions()
                result = whisper.decode(model, mel, options)

                # Write into a text file
                name = RESPATH + "/" + file_name.split('.')[0]
                with open(f"{name}.txt", "w") as f:
                    if result.text:
                        f.write(result.text)
            except RuntimeError:
                pass

if __name__=='__main__':
    get_all_text(FILEPATH, model_type='medium')