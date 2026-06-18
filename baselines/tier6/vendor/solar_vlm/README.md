SolarVLM for Multi-Station PV Power Forecasting

This repository contains the code used for multi-station PV power forecasting in Hebei Province.

The version released here is the one I actually used for experiments based on offline visual features. In other words, the image encoder is run in advance, the visual features are saved as `.npy` files, and training reads these precomputed features directly. The online visual path is still kept in the code, but it is not part of the recommended reproduction path for this release. fileciteturn7file4 fileciteturn7file7

If you want to reproduce the reported results, please follow the offline-feature workflow described below.

1. File placement

The code assumes a directory layout like this:

```text
.
run_pv.py
requirements.txt

tools/
  precompute_vision_feats_qwen3vl.py

exp/
  exp_basic.py
  experiment.py

data_provider/
  data_factory.py
  data_loader_pv.py

layers/
  Embed.py
  GraphLearner.py

utils/
  metrics.py
  tools.py

src/
  SolarVLM/
    model.py
    qwen3_vl_embedding.py
    text_encoders.py
    vision_store.py
```

Before running the project, please place the uploaded files into the corresponding folders.

2. Environment

This project is intended to run on Linux with Python and PyTorch.

If you already exported a txt environment file from the original machine, you can install dependencies with:

```bash
pip install -r requirements.txt
```

The code uses packages such as torch, numpy, pandas, scikit-learn, matplotlib, Pillow, transformers, qwen-vl-utils, and einops. The exact package versions should follow the provided `requirements.txt`. fileciteturn7file2 fileciteturn7file4

3. What data this code expects

The PV data loader does not read one merged csv. It reads eight station files directly from `root_path`, and the station order is fixed as:

```text
station00
station01
station02
station04
station06
station07
station08
station09
```

So the data directory should look like this:

```text
root_path/
  station00.csv
  station01.csv
  station02.csv
  station04.csv
  station06.csv
  station07.csv
  station08.csv
  station09.csv
```

This is the actual organization used by the loader. fileciteturn7file1

For each station csv, the loader looks for the time column in this order:

```text
date_time
date
otherwise the first column
```

Then all timestamps are aligned to a 15-minute grid. The default time range in `run_pv.py` is from `2018-12-01 00:00` to `2019-06-01 00:00`. fileciteturn8file0 fileciteturn7file11

Under the multivariate setting used here, the feature schema is:

```text
nwp_globalirrad
nwp_directirrad
nwp_temperature
nwp_humidity
nwp_windspeed
nwp_winddirection
nwp_pressure
lmd_totalirrad
lmd_diffuseirrad
lmd_temperature
lmd_pressure
lmd_winddirection
lmd_windspeed
power
```

The target is `power`. fileciteturn8file0 fileciteturn7file7

The loader uses chronological splitting. The code fits the scaler only on the training part and then applies it to validation and test data. This is the setting used in the released experiments. fileciteturn8file0

4. Offline visual feature preprocessing

This release uses offline visual features.

The preprocessing script is `tools/precompute_vision_feats_qwen3vl.py`. It reads remote-sensing or cloud images, crops a local region around each PV station using the station longitude and latitude, encodes the cropped images with Qwen3-VL-Embedding-2B, and saves one `.npy` file for each timestamp. If an image cannot be read normally, the script falls back to a black image and records the bad file in `bad_images.txt`. fileciteturn7file6 fileciteturn7file3 fileciteturn7file2

A typical input directory looks like this:

```text
image_root/
  201812010000.png
  201812010010.png
  201812010020.png
  ...
```

A typical output directory looks like this:

```text
vision_feats_qwen3vl/
  201812010000.npy
  201812010010.npy
  201812010020.npy
  ...
```

The script extracts the timestamp key from the digits in the image filename. The saved feature files are then read later by `VisionFeatureStore`. fileciteturn7file2 fileciteturn7file16

In the current code, training reconstructs multi-frame visual context by reading several timestamp files around the target time. So the simplest way to reproduce the released setup is:

```text
precompute with n_frames = 1
train with num_frames = 8
```

This matches the current offline loading path more naturally. fileciteturn7file2 fileciteturn7file7 fileciteturn8file1

Example preprocessing command:

```bash
python tools/precompute_vision_feats_qwen3vl.py \
  --image_root /path/to/all_image \
  --out_dir /path/to/vision_feats_qwen3vl \
  --qwen_path /path/to/QwenQwen3-VL-Embedding-2B \
  --roi_size 128 \
  --n_frames 1 \
  --device cuda \
  --batch_images 32 \
  --normalize 1 \
  --fp16 1
```

5. Training

The main entry is `run_pv.py`.

The defaults in the released code are:

```text
model = SolarVLM
vlm_type = qwen3vl
seq_len = 288
label_len = 144
pred_len = 48
d_model = 128
num_stations = 8
num_frames = 8
gnn_k = 5
learning_rate = 5e-4
batch_size = 16
warmup_epochs = 5
multimodal_epochs = 10
train_epochs = 50
```

The code also supports ablation flags such as `--disable_visual`, `--disable_text`, `--disable_gnn`, and `--disable_cross_site_attn`. fileciteturn8file1

The released training procedure is a three-stage one:

```text
Phase 1: train the temporal backbone
Phase 2: train the multimodal branch
Phase 3: joint fine-tuning
```

This is the actual training logic used in `experiment.py`. fileciteturn0file4

A typical training command is:

```bash
python run_pv.py \
  --is_training 1 \
  --model SolarVLM \
  --data PV \
  --root_path /path/to/pv_csvs \
  --vision_feat_dir /path/to/vision_feats_qwen3vl \
  --qwen3_vl_model_path /path/to/QwenQwen3-VL-Embedding-2B \
  --use_offline_vision \
  --seq_len 288 \
  --label_len 144 \
  --pred_len 48 \
  --patch_len 10 \
  --num_frames 8 \
  --gnn_k 5 \
  --d_model 128 \
  --learning_rate 5e-4 \
  --batch_size 16 \
  --warmup_epochs 5 \
  --multimodal_epochs 10 \
  --train_epochs 50 \
  --gpu 0
```

6. Testing

A typical test command is:

```bash
python run_pv.py \
  --is_training 0 \
  --model SolarVLM \
  --data PV \
  --root_path /path/to/pv_csvs \
  --vision_feat_dir /path/to/vision_feats_qwen3vl \
  --qwen3_vl_model_path /path/to/QwenQwen3-VL-Embedding-2B \
  --use_offline_vision \
  --seq_len 288 \
  --label_len 144 \
  --pred_len 48 \
  --patch_len 10 \
  --num_frames 8 \
  --gnn_k 5 \
  --d_model 128 \
  --gpu 0
```

The test code computes MAE, MSE, RMSE, MAPE, MSPE, and global R2. It also supports inverse transformation back to the original scale when `--inverse` is enabled. fileciteturn0file12 fileciteturn7file9 fileciteturn8file1

7. Output files

Model checkpoints are saved under:

```text
./checkpoints/<setting>/checkpoint.pth
```

Test figures are saved under:

```text
./test_results/<setting>/
```

Prediction arrays and metric files are saved under:

```text
./results/<setting>/
```

These paths come directly from the released training and testing code. fileciteturn7file0 fileciteturn7file9

8. A note about the released scope

To avoid confusion, I want to state this clearly.

What has been used and checked in the released experiments is the offline-visual-feature version. The online visual path is still present in the codebase, but I am not using it as the official reproduction path in this release. If you want to reproduce the current results, please precompute the visual features first and then train with `--use_offline_vision`. fileciteturn7file4 fileciteturn7file16

9. Citation

If you use this repository in your work, please cite the corresponding paper.



10. Contact

If you find a bug or have a question about data formatting, preprocessing, or reproduction, feel free to open an issue.
