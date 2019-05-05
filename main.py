from lab3_tools import *
from lab2_tools import *
from lab1_tools import *
from lab3_proto import *
from lab2_proto import *
from lab1_proto import *

#
phoneHMMs = np.load('lab2_models_all.npz')['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
stateList

#
filename = 'tidigits/disc_4.1.1/tidigits/train/man/nw/z43a.wav'
samples, samplingrate = loadAudio(filename)
lmfcc = mfcc(samples)
wordTrans = list(path2info(filename)[2])
wordTrans

from prondict import prondict
phoneTrans = words2phones(wordTrans, prondict)
phoneTrans
