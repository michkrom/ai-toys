import warnings
warnings.filterwarnings("ignore")

import numpy as np
from torch.utils.data import Dataset, DataLoader
import morseCode
from utils import DatasetFeeder

def morse2List(mcode):
    #   0 0 none
    # . 1 0 dot
    # - 0 1 dash
    v=[]
    for m in mcode:
        v = v + ( [1,0] if m == '.' else [0,1] )
    while len(v) < 2 * 6:
        v = v + [0,0]
    return v
   

# Define some sample data
def genData(dataLen):
    """generate c -> morse traning data"""
    alphabetLen = len(morseCode.morseCode)
    X = np.random.rand(dataLen) * alphabetLen
    X = X // 1 # rounding
    y = [np.zeros(12) for _ in range(len(X))]
    i = 0
    for x in X:
        j = 0
        for c in morseCode.morseCode[int(x)][1]:
            a,b = (1,0) if c == '.' else (0,1)
            y[i][j] = np.float64(a)
            y[i][j+1] = np.float64(b)
            j = j + 2
        i = i + 1
    return X, y

batch_size = 64

X_train, y_train = genData(1000)
train_data = DatasetFeeder(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

X_test, y_test = genData(1000)
test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

def test_train_dataloader():
    for batch, (X, y) in enumerate(train_dataloader):
        print(f"Batch: {batch+1} X: {X.shape} y: {y.shape}")
