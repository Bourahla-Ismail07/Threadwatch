import numpy as np
from threadwatch.metrics import prf,event_recall,evaluate

def test_perfect_and_zero():
    y=np.array([0,1,1,0],dtype=bool)
    assert prf(y,y)[2]==1
    assert prf(np.zeros(4),y)[1]==0

def test_event_recall_counts_event_once():
    y=np.array([0,1,1,1,0,0,1,1,0],dtype=bool)
    p=np.array([0,1,0,1,0,0,1,1,0],dtype=bool)
    assert event_recall(p,y)==1
    assert evaluate(p,y,w=1,s=1)["event_recall"]==1
