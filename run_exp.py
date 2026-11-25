#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from time import time
import traceback
import sys

from river.datasets import synth
from river import drift
from river.base import DriftDetector, DriftAndWarningDetector
from river.preprocessing import StandardScaler
from river.linear_model import ALMAClassifier, LogisticRegression
from river.tree import ExtremelyFastDecisionTreeClassifier, HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier
from river.neighbors import KNNClassifier, SWINN


class RiverStreamWrapper:
    def __init__(self, cls, fun_key, cls_funs, params):
        self.cls = cls
        self.fun_key = fun_key
        self.params = params
        self.cls_funs = cls_funs
        self.classification_function = None
        self.generate_drift()
    def generate_drift(self):
        while True:
            if type(self.cls_funs) is int:
                new_cls_fun = np.random.randint(0,self.cls_funs+1) 
            elif type(self.cls_funs) is list:
                new_cls_fun = self.cls_funs[np.random.randint(0,len(self.cls_funs))]
            else:
                raise ValueError()
            if new_cls_fun != self.classification_function:
                self.classification_function = new_cls_fun
                break
    def take(self, size=1):
        return self.cls(  **({self.fun_key: self.classification_function}|self.params)  ).take(size)

class RandomTree:
    def __init__(self,params):
        self.seed_tree = 42
        self.seed_sample = 42
        self.params = params
        self.generate_drift()
    def generate_drift(self):
        self.seed_tree = (self.seed_tree+np.random.randint(1,100))%2500
        self.seed_sample = (self.seed_sample+np.random.randint(1,100))%2500
    def take(self, size=1):
        return synth.RandomTree(seed_tree=self.seed_tree, seed_sample=self.seed_sample, **self.params).take(size)

class RandomRBF:
    def __init__(self,params):
        self.seed_model = 42
        self.seed_sample = 42
        self.params = params
        self.generate_drift()
    def generate_drift(self):
        self.seed_model = (self.seed_model+np.random.randint(1,100))%2500
        self.seed_sample = (self.seed_sample+np.random.randint(1,100))%2500
    def take(self, size=1):
        return synth.RandomRBF(seed_model=self.seed_model, seed_sample=self.seed_sample, **self.params).take(size)

generators = {"Agrawal": lambda param: RiverStreamWrapper(synth.Agrawal,"classification_function",9,param), 
              "SEA": lambda param: RiverStreamWrapper(synth.SEA,"variant",[0,1,2,3],param), 
              "Sine": lambda param: RiverStreamWrapper(synth.Sine,"classification_function",3,param),
              "RandomTree":lambda param: RandomTree(param),
              "RandomRBF":lambda param: RandomRBF(param)
             }

class LogDriftDetector(DriftAndWarningDetector):
    def __init__(self, inner_detector):
        super().__init__()
        self.inner_detector = inner_detector
        self.reset() 

    def reset(self):
        self.working_detector = self.inner_detector.clone()
        self.detect_log = []
        self.warn_log = []
        self.pos = -1

    def update(self, x):
        self.working_detector.update(x)
        self.pos += 1

        if hasattr(self.working_detector, "_warning_detected"):
            self._warning_detected = self.working_detector._warning_detected
            if self.working_detector._warning_detected:
                self.warn_log.append(self.pos)

        self._drift_detected = self.working_detector._drift_detected
        if self.working_detector._drift_detected:
            self.detect_log.append(self.pos)

class GTDetector(DriftDetector):
    def __init__(self, drifts):
        super().__init__()
        self.drifts = drifts
        self.reset() 
    def reset(self):
        self.pos = -1
    def update(self, x):
        self.pos+=1
        self._drift_detected = self.pos in self.drifts

class DummyDetector(DriftDetector):
    def __init__(self, dist=None, p=None):
        super().__init__()
        assert (dist is not None) or (p is not None)
        assert (dist is None) or (type(dist) is int and dist > 0)
        assert (p is None) or (type(p) is float and 0 <= p and p <= 1)
        self.dist=dist
        self.p=p
        self.reset() 
    def reset(self):
        self.pos = -1
        if self.dist is not None:
            self.offset = np.random.randint(0,self.dist)
    def update(self, x):
        self.pos+=1
        if self.dist is not None:
            self._drift_detected = (self.pos+self.offset) % self.dist == 0
        elif self.p is not None:
            self._drift_detected = np.random.random() < self.p
        else:
            raise ValueError()

def generate_data(generator, params={}, warmup=350, cooldown=350, drift_area=100, max_window_margin=100):
    gen = generators[generator](params)
    gen.generate_drift()
    drift_point = np.random.randint(0,drift_area)
    before_drift = warmup+max_window_margin+drift_point
    after_drift  = (drift_area-drift_point)+max_window_margin+cooldown
    assert before_drift + after_drift == warmup + max_window_margin + drift_area + max_window_margin + cooldown
    stream = list(gen.take(before_drift))
    gen.generate_drift()
    stream += list(gen.take(after_drift))

    return stream, before_drift


def get_detector(name, alpha, drifts):
    if name=="NoDrift":
        assert alpha is None
        return ("NoDrift",{},DummyDetector(p=0.))
    elif name=="GroundTrue":
        assert alpha is None
        return ("GroundTruth",{},GTDetector(drifts))
    elif name == "DummyRand":
        assert 0 < alpha and alpha < 1
        return ("DummyRandom",{"alpha":alpha},DummyDetector(p=alpha))
    elif name == "DummyDist":
        assert 0 < alpha and alpha < 1
        # want E[Bin(alpha,length)] = length / window
        # alpha * length = length / window
        dist = round(1/alpha)
        return ("DummyDistance",{"alpha": alpha, "distance": dist},DummyDetector(dist=dist))
    elif name == "DDM":
        assert 0 < alpha and alpha < 1
        alpha1 = alpha/3*2
        theta0 = norm.ppf(1-alpha/2)
        theta1 = norm.ppf(1-alpha1/2)
        return ("DDM",{"alpha": alpha, "warning_threshold": theta1, "drift_threshold": theta0},drift.binary.DDM(drift_threshold=theta0, warning_threshold=theta1))
    elif name in "EDDM":
        assert 0 < alpha and alpha < 1
        warn=alpha/0.99*0.95
        return ("EDDM",{"alpha": alpha, "warn": warn},drift.binary.EDDM( beta=1-alpha, alpha=1-warn ))
    elif name in "FHDDM":
        assert 0 < alpha and alpha < 1
        return ("FHDDM",{"alpha": alpha},drift.binary.FHDDM( confidence_level=alpha ))
    elif name in "HDDM_A":
        assert 0 < alpha and alpha < 1
        warn=alpha/(1-0.001)*(1-0.005)
        return ("HDDM_A",{"alpha": alpha, "warning_confidence": warn, "drift_confidence": alpha},drift.binary.HDDMA( warning_confidence=warn, drift_confidence=alpha ))
    elif name in "HDDM_W":
        assert 0 < alpha and alpha < 1
        warn=alpha/(1-0.001)*(1-0.005)
        return ("HDDM_W",{"alpha": alpha, "warning_confidence": warn, "drift_confidence": alpha},drift.binary.HDDMW( warning_confidence=warn, drift_confidence=alpha ))
    elif name in "ADWIN":
        assert 0 < alpha and alpha < 1
        return ("ADWIN",{"alpha": alpha},drift.ADWIN( delta=alpha ))
    elif name in "KSWIN":
        assert 0 < alpha and alpha < 1
        return ("KSWIN",{"alpha": alpha},drift.KSWIN( alpha=alpha ))
    else:
        raise ValueError(f"{name} not found")

run_id = int(sys.argv[1]) if len(sys.argv) == 2 else -1
print("Running experiment id: ",run_id)

n_itr = 50
deltas = [0.05,0.01,0.005,0.001,0.0005,0.0001]
detectors = [(detector_name,delta) for delta in deltas for detector_name in ["DummyRand","DummyDist","DDM","EDDM","FHDDM","HDDM_A","HDDM_W","ADWIN","KSWIN"]]+[("NoDrift",None),("GroundTrue",None)]
models = [("kNN", StandardScaler() | KNNClassifier(engine=SWINN(maxlen=200))),
          ("HT", HoeffdingTreeClassifier()),
          ("HAT",HoeffdingAdaptiveTreeClassifier()),
          ("EFDT", ExtremelyFastDecisionTreeClassifier()),
          ("ALMA", StandardScaler() | ALMAClassifier()),
          ("LR", StandardScaler() | LogisticRegression())]

streams = [("SEA",{}),("Sine",{}),("Agrawal",{}),("RandomTree",{}),("RandomRBF",{})]
results = []

save_time = time()
for exp_id,(stream_name,stream_params) in enumerate(tqdm(n_itr*streams,delay=60)):
    stream_instance,drift_pos = generate_data(stream_name,stream_params)
    for detector_name,delta in detectors:
        for model_name,model_mdl in models:
            detector_name_out,detector_params_out,detector = get_detector(detector_name,delta,[drift_pos])
            detector_log = LogDriftDetector(detector)
            stream_model = drift.DriftRetrainingClassifier(
                train_in_background=False,
                model=model_mdl.clone(),
                drift_detector=detector_log
            )

            start_time=time()
            yout, pout, error = [], [], []
            for x,y in stream_instance[:]:
                yhat = stream_model.predict_one(x)
                if yhat is not None:
                    yout.append( yhat )
                    error.append( y!=yhat )
                if hasattr(model_mdl, "predict_proba_one"):
                    pyhat = stream_model.predict_proba_one(x)
                    if pyhat is not None:
                        pout.append(pyhat)
                stream_model.learn_one(x,y)
            end_time=time()

            results.append({"exp_id":(run_id,exp_id),
                            "stream name":stream_name,"stream params":stream_params, 
                            "model":model_name,
                            "detector":detector_name_out,"detector params":detector_params_out,
                            "stream":stream_instance, 
                            "drift position":drift_pos, 
                            "pred":yout, 
                            "pred_proba":pout, 
                            "err":error, 
                            "detection log":detector_log.detect_log, 
                            "warning log":detector_log.warn_log,
                            "time":end_time-start_time})
            if time()-save_time > 10*60:
                try:
                    pd.DataFrame(results).to_pickle(f"out/run_{run_id}.pkl.xz")
                    save_time = time()
                except KeyboardInterrupt as ex:
                    raise ex
                except Exception as ex:
                    traceback.print_exc(file=sys.stdout)
while True:
    try:
        pd.DataFrame(results).to_pickle(f"out/run_{run_id}.pkl.xz")
        break
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        time.sleep(1)


