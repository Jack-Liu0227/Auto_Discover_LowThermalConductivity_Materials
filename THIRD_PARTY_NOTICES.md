# Third-Party Notices

This file summarizes important third-party components used or integrated by ADLM. ADLM's repository-level MIT License applies only to the original ADLM source code and documentation. It does not relicense third-party code, model weights, datasets, manuscripts, generated artifacts, or services.

## MatterSim

- Upstream project: [microsoft/mattersim](https://github.com/microsoft/mattersim)
- Copyright holder: Microsoft Corporation
- Upstream license: MIT License
- Use in ADLM: MatterSim is used as an external Python dependency through `mattersim>=1.2.0` and wrapped by `src/tools/mattersim_wrapper.py`. ADLM does not vendor MatterSim source code.
- Model note: MatterSim provides pretrained `MatterSim-v1` checkpoints such as `MatterSim-v1.0.0-1M` and `MatterSim-v1.0.0-5M`. Reports and papers using MatterSim-backed results should identify the exact model version and checkpoint used, for example `MatterSim-v1.0.0-1M`.

MatterSim citation:

```bibtex
@article{yang2024mattersim,
  title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
  author={Han Yang and Chenxi Hu and Yichi Zhou and Xixian Liu and Yu Shi and Jielan Li and Guanzhi Li and Zekun Chen and Shuizhou Chen and Claudio Zeni and Matthew Horton and Robert Pinsler and Andrew Fowler and Daniel Zügner and Tian Xie and Jake Smith and Lixin Sun and Qian Wang and Lingyu Kong and Chang Liu and Hongxia Hao and Ziheng Lu},
  year={2024},
  eprint={2405.04967},
  archivePrefix={arXiv},
  primaryClass={cond-mat.mtrl-sci},
  url={https://arxiv.org/abs/2405.04967},
  journal={arXiv preprint arXiv:2405.04967}
}
```

## CrystaLLM

- Upstream project: [lantunes/CrystaLLM](https://github.com/lantunes/CrystaLLM)
- Copyright holder: Luis M. Antunes
- Upstream code license: MIT License
- Use in ADLM: CrystaLLM code and wrappers are integrated under `src/tools/crystallm/` and `src/tools/crystallm_wrapper.py` for crystal-structure generation.
- Model/data/artifact note: CrystaLLM trained models, training sets, and model-generated artifacts are published through Zenodo and are released under the Creative Commons Attribution 4.0 International License (CC-BY 4.0). ADLM does not relicense those model weights, datasets, or generated artifacts.

CrystaLLM citation:

```bibtex
@article{antunes2024crystal,
  title={Crystal structure generation with autoregressive large language modeling},
  author={Antunes, Luis M and Butler, Keith T and Grau-Crespo, Ricardo},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={10570},
  year={2024},
  publisher={Nature Publishing Group}
}
```

## Practical Guidance

- Keep ADLM's original code citation and license separate from MatterSim and CrystaLLM citations.
- When publishing results that depend on MatterSim, report the exact MatterSim model version and checkpoint.
- When publishing or redistributing CrystaLLM model outputs, datasets, or model weights, preserve the relevant CC-BY 4.0 attribution requirements.
- If additional third-party tools or model weights are added later, update this file before redistribution.
