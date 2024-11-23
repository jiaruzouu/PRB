## Baseline Running
**Prepare Datasets**
```
python ogbdataset.py
```

**baseline running**

Command options for different baselines.

| name     | $model name    | command change     |
|----------|-----------|--------------------|
| GAE      | cn0       |                    |
| NCN      | cn1       |                    |
| NCNC     | incn1cn1  |                    |
| NCNC2    | incn1cn1  | add --depth 2  --splitsize 131072    |
| GAE+CN   | scn1      |                    |
| NCN2     | cn1.5     |                    |
| NCN-diff | cn1res    |                    |
| NoTLR    | cn1       | delete --maskinput |

The scipts for running three ogbd datasets
e.g. Collab
```
python baseline_run.py  --xdp 0.25 --tdp 0.05 --pt 0.1 --gnnedp 0.25 --preedp 0.0 --predp 0.3 --gnndp 0.1  --probscale 2.5 --proboffset 6.0 --alpha 1.05  --gnnlr 0.0082 --prelr 0.0037  --batch_size 65536  --ln --lnnn --predictor $model --dataset collab  --epochs 100 --runs 10 --model gcn --hiddim 64 --mplayers 1  --testbs 131072  --maskinput --use_valedges_as_input   --res  --use_xlin  --tailact 
```
