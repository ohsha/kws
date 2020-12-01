from scipy.io import wavfile
import speech_recognition as sr
from tqdm import tqdm
import pandas as pd
import numpy as np
import os



'''
Use the main function 'final_creator' in order to get output of 3 lists:  train_data, test_data, val_data.
Each list contain 2 arrays = [x,y]
The function argument:
- main_path - library path of the audio files folder
- classes_list - the list of words you need to classify
'''

## deciding whether the audio samples are definitely speech or not
def speech_recognition(samples,rate=16000):
    is_it_speech = 0
    recognizer = sr.Recognizer()
    harvard = sr.AudioData(samples.tobytes(), sample_rate=rate, sample_width=samples.dtype.itemsize)
    text_output = recognizer.recognize_google(harvard, show_all=True)
    if len(text_output) != 0:
        is_it_speech = 1
    return is_it_speech

## padding the audio samples to be 16000 length
def audio_padding(samples):
    samples = np.float16(samples)
    if len(samples)<=16000:
        temp_vec = np.zeros(16000, dtype=int)
        temp_vec[-len(samples):] = samples[:16000] / 256.0
        return temp_vec
    else:
        return samples[-16000:]

###### creating the speech data only
def create_speech_data(main_path,classes_list):
    wav_list,labels,file_name_list = [],[],[]
    for class_i in tqdm(classes_list):
        file_list = os.listdir(main_path + '/' + class_i)
        temp_a = pd.Series(file_list)
        temp_b = temp_a.str.split("_", expand=True)[2].str.split('.', expand=True)
        final_file_list = temp_a[temp_b[1] == 'wav'].values
        for file_i in tqdm(final_file_list):
            wav_path = main_path + '/' + class_i + '/' + file_i
            _, samples = wavfile.read(wav_path)
            result = speech_recognition(samples)
            if result==1:
                wav_list += [samples]
                labels += [class_i]
                file_loc = np.where(np.array(file_list) == file_i)[0][0]
                temp_file_list = file_list[file_loc:(file_loc+18)]
                file_name_list+= temp_file_list
                for file in tqdm(temp_file_list[1:]):
                    wav_path = main_path + '/' + class_i + '/' + file
                    _, samples = wavfile.read(wav_path)
                    samples = audio_padding(samples)
                    wav_list += [samples]
                    labels += [class_i]
    return np.expand_dims(np.array(wav_list), -1), np.array(labels) , np.array(file_name_list)

## creating the background noise data
def create_background_noise(main_path):
    noise_path = main_path + "/noise"
    noise_folder_list = os.listdir(noise_path)
    noise_wav_list = []
    for folder_i in tqdm(noise_folder_list):
        noise_file_list = os.listdir(noise_path+'/'+folder_i)
        for file_i in tqdm(noise_file_list):
            _, samples = wavfile.read(noise_path+ '/' + folder_i + '/'+file_i)
            samp_bins = audio_padding(samples)
            noise_wav_list += [samp_bins]
    return np.expand_dims(np.array(noise_wav_list), -1)

# spliting the data to test and train
def split_data(x_speech,y_speech,x_noise,speech_list,train_percentage=0.8,test_percentage = 0.1):

    # speech data
    sorted_loc= np.argsort(speech_list)
    train_s_index,test_s_index = int(len(speech_list) * train_percentage), int(len(speech_list) * test_percentage)
    train_loc , test_loc , val_loc = np.split(sorted_loc, [train_s_index,test_s_index])
    x_s_train,x_s_test,x_s_val = x_speech[train_loc], x_speech[test_loc], x_speech[val_loc]
    y_s_train,y_s_test,y_s_val = y_speech[train_loc], y_speech[test_loc], y_speech[val_loc]

    # noise data
    np.random.shuffle(x_noise)
    train_n_index,test_n_index = int(len(x_noise) * train_percentage), int(len(x_noise) * test_percentage)
    x_n_train,x_n_test,x_n_val = x_noise[:train_n_index], x_noise[train_n_index:test_n_index], x_noise[test_n_index:]

    x_train = np.concatenate([x_s_train, x_n_train], axis=0)
    y_train = np.concatenate([y_s_train, np.repeat('background_noise',len(x_n_train))], axis=0)

    x_test = np.concatenate([x_s_test, x_n_test], axis=0)
    y_test = np.concatenate([y_s_test, np.repeat('background_noise',len(x_n_test))], axis=0)

    x_val = np.concatenate([x_s_test, x_n_test], axis=0)
    y_val = np.concatenate([y_s_test, np.repeat('background_noise',len(x_n_test))], axis=0)

    return [x_train,y_train], [x_test,y_test], [x_val,y_val]

### main function to create the data
def final_creator(main_path,classes_list, train_percentage=0.8, test_percentage=0.1):
    x_speech, y_speech ,speech_list = create_speech_data(main_path,classes_list)
    x_noise = create_background_noise(main_path)
    train, test, val = split_data(x_speech, y_speech, x_noise, speech_list, train_percentage, test_percentage)
    return train, test, val

#######################################################################################


if __name__ == '__main__':

    '''
    Example of Use : 
    
    '''
    classes_list = ['down', 'go', 'left', 'no', 'off','on', 'right',  'stop', 'up','yes']
    main_path = r''

    train_data , test_data , val_data = final_creator(main_path,classes_list)
    export_path = r''
    np.save(file=os.path.join(export_path, 'train.npy'), arr=train_data)
    np.save(file=os.path.join(export_path, 'test.npy'), arr=test_data)
    np.save(file=os.path.join(export_path, 'val.npy'), arr=val_data)