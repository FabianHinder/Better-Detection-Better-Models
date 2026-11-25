# Does Better Drift Detection Yield Better Models? An Empirical Study

Experimental code of paper

* [[1]](#paper_ESANN) [_"Does Better Drift Detection Yield Better Models? An Empirical Study"_](TODO) 
* [[2]](#paper_ArXiv) [(extended ArXiv version)](TODO) 

by Valerie Vaquet and Fabian Hinder.

## Abstract

Stream learning algorithms often rely on drift detectors to trigger model updates, assuming accurate detection ensures high predictive performance. However, the interplay between detector quality and model accuracy remains underexplored. This work empirically investigates this relationship using synthetic datasets with real drift. We evaluate four hypotheses regarding the impact of correct detections, false alarms, detection delay, and repeated updates on the model error. Our results reveal that drift detection is beneficial only in scenarios with severe drift. In milder drift settings, false alarms and excessive updates can degrade performance. We conclude that the optimal strategy depends heavily on drift severity, and find that oftentimes relying on passive adaption schemes is a valid option.

## Requirements

* Python 
* Numpy, SciPy, Pandas, Matplotlib
* tqdm
* River

## Data and Usage

Run `run_exp.py` with an optional experimental id (integer) as argument. The results will be saved automatically. 

## Cite

Cite our papers:

<a name="paper_ESANN">[1]</a> V. Vaquet, F. Hinder [_"Does Better Drift Detection Yield Better Models? An Empirical Study"_](TODO) [(extended ArXiv version)](TODO) 
```
IN REVIEW
}
```

```
@article{vasquez_better_2026,
author={Valerie Vaquet. and Fabian Hinder.},
title={{Does Better Drift Detection Yield Better Models? An Empirical Study},
year={2026},
}
```

## License

This code has a MIT license.


