from lab3_tools import *
from lab2_tools import *
from lab1_tools import *
from lab3_proto import *
from lab2_proto import *
from lab1_proto import *
from prondict import prondict
import os
import pdb
from tqdm import tqdm


def task(task):
    # ----- 4.1 -------
    phoneHMMs = np.load('lab2_models_all.npz')['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]

     # ----- 4.2 -------
    filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
    samples, samplingrate = loadAudio(filename)
    lmfcc = mfcc(samples)
    wordTrans = list(path2info(filename)[2])#sequence of digits (word level transcription)
    phoneTrans = words2phones(wordTrans, prondict)#phone level transcription
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                  for stateid in range(nstates[phone])]
    viterbiStateTrans = forcedAlignment(lmfcc, utteranceHMM, stateTrans)
    # pdb.set_trace()
    frames2trans(viterbiStateTrans, outfilename='z43a.lab')

    if task == '4.3':
        traindata = []
        #for root, dirs, files in os.walk('tidigits/disc_4.1.1/tidigits/train'):
        for root, dirs, files in os.walk('tidigits/disc_4.2.1/tidigits/test'):
            for file in files:
                if file.endswith('.wav'):
                    filename = os.path.join(root, file)
                    samples, samplingrate = loadAudio(filename)
                    lmfcc_ = mfcc(samples, samplingrate=samplingrate)
                    mspec_ = mspec(samples, samplingrate=samplingrate)
                    wordTrans = list(path2info(filename)[2])
                    phoneTrans = words2phones(wordTrans, prondict)
                    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
                    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans
                                  for stateid in range(nstates[phone])]
                    viterbiStateTrans = forcedAlignment(lmfcc, utteranceHMM, stateTrans)
                    targets = [stateList.index(s) for s in viterbiStateTrans]
                    traindata.append({'filename': filename, 'lmfcc': lmfcc_, 'mspec': mspec_, 'targets': targets})
        #np.savez('traindata.npz', traindata=traindata)
        np.savez('testdata.npz', traindata=traindata)




if __name__ == '__main__':
    task('4.3')

