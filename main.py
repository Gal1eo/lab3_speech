from lab3_tools import *
from lab2_tools import *
from lab1_tools import *
from lab3_proto import *
from lab2_proto import *
from lab1_proto import *
from prondict import prondict
import os


if __name__ == '__main__':
    #4.1
    phoneHMMs = np.load('lab2_models_all.npz')['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

    # 4.2
    filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
    samples, samplingrate = loadAudio(filename)
    lmfcc = mfcc(samples)
    wordTrans = list(path2info(filename)[2])
    phoneTrans = words2phones(wordTrans, prondict)
    viterbiState = forcedAlignment(lmfcc, phoneHMMs, phoneTrans)

    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans \
                  for stateid in range(nstates[phone])]
    # TODO viterbiStateTrans = stateTrans viterbiState

    frames2trans(viterbiStateTrans, outfilename='z43a.lab')


    # 4.3
    traindata = []
    for root, dirs, files in os.walk('tidigits/disc_4.1.1/tidigits/train'):
        for file in files:
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
    samples, samplingrate = loadAudio(filename)
    #  TODO   for feature extraction and forced alignment
    traindata.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': 'mspec', 'targets': targets})
    np.savez('traindata.npz', traindata=traindata)
    # same witht test files tidigits/disc_4.2.1/tidigits/test

