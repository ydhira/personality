import torchaudio 
import yaml
import glob
import os 
import pandas as pd 
import torchaudio.transforms as T

def audio_chunk(audiopath, chunking_length=10):
    '''
    Given an audiopath and the chunking_length
    return different waveforms of the length of chunking_length.
    Default for chunking length is 10 secs. The remaining of the 
    audio is also returned

    Input: 
        1. audiopath (str): path of the audio file
        2. chunking_length (int): in seconds. length of the chunks 
    Return:
        1. List: list of waveforms 
    '''
    waveform, sample_rate = torchaudio.load(audiopath)
    frame_offset = 0 
    num_frames = sample_rate * chunking_length # 1 sec = sample_rate frames, chunking_length sec = X frames 
    results = []
    while (frame_offset < waveform.shape[1]):
        waveform1 = waveform[:, frame_offset : frame_offset + num_frames]
        frame_offset = frame_offset + num_frames
        results.append(waveform1)
    return results 


def parse_data(indir, labelfile, metafile):
    '''
    Given the indirectory and labelfile, this function 
    chunks the audio into multiple chunks

    Input:
        1. indir (str): where the audio files are present 
        2. labelfile (str): path to the csv file where the OCEAN traits is present for presidents
        3. metafile (str): path to the csv file where all the president meta 
            information is present 

    Return:
        1. list: containing waveforms of the audio path 
        2. tuple: (list of O, C, E, A, N, names, period)
    '''
    labeldb = pd.read_csv(labelfile)
    metadb = pd.read_csv(metafile)

    wavfiles = glob.glob(os.path.join(indir, '*.wav'))
    waveforms = []
    president_names = []
    periods = []
    labels = []
    for wavfile in wavfiles:
        res = audio_chunk(wavfile)
        waveforms.extend(res)
        num_chunks = len(res)

        row = metadb.loc[metadb['Audio FileName']==os.path.basename(wavfile)]
        president_name = row['President'].item()
        presidential_period = row['Presidential Period'].item()
        ocean_db = labeldb.loc[labeldb['Name']==president_name]
        O = ocean_db['O'].item()
        C = ocean_db['C'].item()
        E = ocean_db['E'].item()
        A = ocean_db['A'].item()
        N = ocean_db['N'].item()

        president_names.extend([president_name]*num_chunks)
        periods.extend([presidential_period]*num_chunks)
        labels.extend([{'O':O, 'C':C, 'E':E, 'A':A, 'N':N}]*num_chunks)    
    
    return waveforms, (labels, president_names, periods)

def get_mfcc(mfcc_transform, waveforms):
    '''
    Given a list of raw audio, this function returns the 
    mfcc features for them 

    Input:
        1. mfcc_transform: MFCC object from pytorch 
        2. waveforms (list): List of raw waveforms where ech waveform 
            is of the shape 1, X 
    
    Return:
        1. list: list of mfcc 
    '''
    mfccs = []
    for waveform in waveforms:
        mfcc = mfcc_transform(waveform)
        mfccs.append(mfcc)
    return mfccs

def get_mfcc_object(args):
    mfcc_transform = T.MFCC(
    	sample_rate=args['sample_rate'],
    	n_mfcc=args['mfcc']['n_mfcc'],
    	melkwargs={
      	    'n_fft': args['mfcc']['n_fft'],
      	    'n_mels': args['mfcc']['n_mels'],
      	    'hop_length': args['mfcc']['hop_length'],
      	    'mel_scale': args['mfcc']['mel_scale'],
   	   }
	)
    return mfcc_transform 

stream = open('config.yaml', 'r')
args = yaml.safe_load(stream)
chunking_length = args['chunking_length']
sample_rate = args['sample_rate']
print(args)
waveforms, (labels, president_names, periods) = parse_data('../president_data/WBush/', '../president_personality.csv', '../all_metadata.csv')
mfcc_transform = get_mfcc_object(args)
mfccs = get_mfcc(mfcc_transform, waveforms)
import pdb 
pdb.set_trace()
