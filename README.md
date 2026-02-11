
# FramePack

Official implementation and desktop software for ["Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models"](https://lllyasviel.github.io/frame_pack_gitpage/).

Links: [**Paper**](https://arxiv.org/abs/2504.12626), [**Project Page**](https://lllyasviel.github.io/frame_pack_gitpage/)

FramePack is a next-frame (next-frame-section) prediction neural network structure that generates videos progressively. 

FramePack compresses input contexts to a constant length so that the generation workload is invariant to video length.

FramePack can process a very large number of frames with 13B models even on laptop GPUs.

FramePack can be trained with a much larger batch size, similar to the batch size for image diffusion training.

my re

# Cite

    @inproceedings{zhang2025framepack,
        title={Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models},
        author={Lvmin Zhang and Shengqu Cai and Muyang Li and Gordon Wetzstein and Maneesh Agrawala},
        booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
        year={2025},
    }

    @article{zhang2025framepackv1,
        title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},
        author={Lvmin Zhang and Maneesh Agrawala},
        journal={Arxiv},
        year={2025}
    }
