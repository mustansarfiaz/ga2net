# GA2-Net: Guided-Attention and Gated-Aggregation Network for Medical Image Segmentation

## News
-Accepted in Pattern Recognition🥳
- [Paper](https://www.sciencedirect.com/science/article/pii/S0031320324005636)

## Highlights
-----------------
- We propose GA2Net to capture complex shapes of the tissues for better segmentation.
- Our HGFA enhances the most relevant feature information for pixel-precise segmentation using deep supervision.
- Adaptive aggregation adjusts the receptive fields for each stage feature.
- Our MGFA modules in the decoder are crucial to obtaining accurate boundaries of the tissue objects.

Introduction
-----------------
GA2Net is a network consisting of an encoder, bottleneck, and decoder designed for effective feature extraction and segmentation. 
The encoder captures multi-scale features, while the bottleneck employs a hierarchical-gated feature aggregation (HGFA) mechanism to enhance spatial understanding. 
Adaptive aggregation (AA) in the decoder dynamically adjusts receptive fields, replacing traditional skip connections to better capture contextual information. 
Additionally, mask-guided feature attention (MGFA) modules use foreground priors to emphasize important structural details. 
Intermediate supervision is applied in both the bottleneck and decoder to improve boundary detection and tissue localization.

## Contact
If you have any questions, please create an issue on this repository or contact us at mustansar.fiaz@mbzuai.ac.ae

<hr />

## References
Our code is based on [SA-Net](https://github.com/mustansarfiaz/SA2-Net), [Awesome-U-Net](https://github.com/NITR098/Awesome-U-Net),  [UCTransNet](https://github.com/McGregorWwww/UCTransNet), and [CASCADE](https://github.com/SLDGroup/CASCADE/tree/main)   repositories. We thank them for releasing their baseline code.

For ISIC2018 and SegPC2021, we follow [Awesome-U-Net](https://github.com/NITR098/Awesome-U-Net).
For ACDC and polyps, we follow [CASCADE](https://github.com/SLDGroup/CASCADE/tree/main).

* **GA2-Net**: "Guided-attention and gated-aggregation network for medical image segmentation", PR, 2024 (*MBZUAI*). [[Paper]([https://arxiv.org/abs/2309.16661)][[PyTorch](https://github.com/mustansarfiaz/SA2-Net](https://www.sciencedirect.com/science/article/pii/S0031320324005636))]




- ## Citation

```
@article{fiaz2024guided,
  title={Guided-attention and gated-aggregation network for medical image segmentation},
  author={Fiaz, Mustansar and Noman, Mubashir and Cholakkal, Hisham and Anwer, Rao Muhammad and Hanna, Jacob and Khan, Fahad Shahbaz},
  journal={Pattern Recognition},
  volume={156},
  pages={110812},
  year={2024},
  publisher={Elsevier}
}

@inproceedings{fiaz2022sat,
  title={SA2-Net: Scale-aware Attention Network for Medical Image Segmentation},
  author={Fiaz, Mustansar and Heidari, Moein and Anwar, Rao Muhammad and Cholakkal, Hisham},
  booktitle={BMVC},
  year={2023}
}
