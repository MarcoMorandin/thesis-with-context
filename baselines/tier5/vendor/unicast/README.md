# UniCast: A Unified Multimodal Prompting Framework for Time Series Forecasting
Implementation of multimodal time-series forecasting framework in [UniCast: A Unified Multimodal Prompting Framework for Time Series Forecasting](http://arxiv.org/abs/2508.11954)

<div align="center">
      <p>
        <strong>Sehyuk Park</strong><sup>1</sup>,
        <strong>Soyeon Caren Han</strong><sup>1, 2</sup>
        <strong>Eduard Hovy</strong><sup>2</sup>
      </p>
</div>

<div align="center">
    <p>
        <sup>1</sup> Pohang University of Science and Technology
        <sup>2</sup> The University of Melbourne
    </p>
</div>

<div align="center">
<p>
      <sup>1</sup> <a href="mailto:percy212@postech.ac.kr,">percy212@postech.ac.kr</a>,  
      <a href="mailto:drcarenhan@postech.ac.kr">drcarenhan@postech.ac.kr</a> 
      <sup>2</sup> <a href="mailto:caren.han@unimelb.edu.au">caren.han@unimelb.edu.au</a>,  
      <a href="mailto:eduard.hovy@unimelb.edu.au">eduard.hovy@unimelb.edu.au</a>
</p>
</div>

![Model Figure](./figures/Model_Structure.png)
## Requirements
This project leverages two Time-Series Foundation Models: **Timer** and **Chronos**.  
Each model requires a separate Python environment:

- **Timer**: `python==3.10.16`
- **Chronos**: `python==3.11.11`

Other dependencies can be installed from the corresponding `requirements.txt` file for each model.

### Environment Setup

**Timer:**
```bash
conda create -n timer python=3.10.16
conda activate timer
pip install -r requirements/timer_requirements.txt
```

**Chronos:**

```bash
conda create -n chronos python=3.11.11
conda activate chronos
pip install -r requirements/chronos_requirements.txt
```

## Dataset Preparation
We use a subset of the evaluation dataset from **Chronos**.  
- All CSV files are stored in the `csv/` folder.  
- The `dataset/` folder contains a `create_dataset.py` script for each dataset.  

For converting time-series data into images, we follow the plotting approach used in **ViTST**.  

To generate the datasets, simply run:
```bash
cd dataset
bash create_dataset.sh
```
## Pretrained Models
UniCast utilizes:  
- **Time-Series Models**: Timer, Chronos  
- **Vision Encoders**: CLIP, BLIP  
- **Text Encoders**: Qwen, LLaMA  

Each model requires its corresponding pretrained configuration and weights.  
For each model, a `save_pretrained_model.py` script is provided in its respective folder.  

To download and save all pretrained models, simply run:
```bash
cd models
bash save_pretrained_model.sh
```

## Run
For each TSFM, separate shell scripts are provided for **training** and **testing**.  
These scripts are configured to iterate over different combinations of **vision encoders** and **text encoders**.

- To train:
```bash
# For Timer
bash train_multi_modal_timer.sh

# For Chronos
bash train_multi_modal_chronos.sh
```
- To evaluate:

```bash
# For Timer
bash test_multi_modal_timer.sh

# For Chronos
bash test_multi_modal_chronos.sh
```

## Evaluation Results
![Result](./figures/Results.jpg)
When compared with six baseline models, **UniCast** achieved better performance in a parameter-efficient manner while keeping the backbone frozen.
![Ablation](./figures/Ablation.jpg)
Incorporating either visual or textual context improves performance over the time-series-only model, while combining both modalities consistently yields the best results.
## Qualitative Analysis
![Qualitative Analysis](./figures/qa.png)
The figure compares four configurations: **TSFM Zero-Shot**, **TSFM with Prompt Tuning**, **TSFM with Vision Encoder**, and **TSFM with both Vision and Text Encoders**. 
It shows that adding more modalities enables the model to capture patterns more effectively.

## Citation
If you find our UniCast framework helpful, we would appreciate it if you could cite our paper.
```bibtex
@misc{park2025unicastunifiedmultimodalprompting,
      title={UniCast: A Unified Multimodal Prompting Framework for Time Series Forecasting}, 
      author={Sehyuk Park and Soyeon Caren Han and Eduard Hovy},
      year={2025},
      eprint={2508.11954},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.11954}, 
}
```