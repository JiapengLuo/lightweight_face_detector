
# Evaluation code for lightweight face detection

We provide 4 different setting checkpoints.

## Requirements

1. tensorflow == 1.12
2. Pillow
3. scipy
4. numpy

change the path in eval_fddb.py and eval_wider.py to customed path

### results of the enable_hflip

|            | WIDERFACE easy | WIDERFACE medium | WIDERFACE hard |
|------------|----------------|------------------|----------------|
| val (mAP)  | 0.899          | 0.878            | 0.750          |
| test (mAP) | 0.871          | 0.873            | 0.780          |

|             |     FDDB       |
|-------------|----------------|
| recall@2000 | 0.979          |

