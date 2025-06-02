# **Distilling CLIP with Dual Guidance for Learning Discriminative Human Body Shape Representation**

CVPR 2024.[PDF](https://cvlab.cse.msu.edu/pdfs/Liu_Kim_Ren_Liu_CLIP3DREID.pdf), [Supp](https://cvlab.cse.msu.edu/pdfs/Liu_Kim_Ren_Liu_CLIP3DREID_supp.pdf), [Project](https://cvlab.cse.msu.edu/project-clip3dreid.html)]

**Feng Liu, Minchul Kim, Anil Jain, Xiaoming Liu**

Person Re-Identification (ReID) holds critical importance in computer vision with pivotal applications in public safety and crime prevention. Traditional ReID methods, reliant on appearance attributes such as clothing and color, encounter limitations in long-term scenarios and dynamic environments. To address these challenges, we propose CLIP3DReID, an innovative approach that enhances person ReID by integrating linguistic descriptions with visual perception, leveraging pretrained CLIP model for knowledge distillation. Our method first employs CLIP to automatically label body shapes with linguistic descriptors. We then apply optimal transport theory to align the student model's local visual features with shape-aware tokens derived from CLIP's linguistic output. Additionally, we align the student model's global visual features with those from the CLIP image encoder and the 3D SMPL identity space, fostering enhanced domain robustness. CLIP3DReID notably excels in discerning discriminative body shape features, achieving state-of-the-art results in person ReID. Our approach represents a significant advancement in ReID, offering robust solutions to existing challenges and setting new directions for future research.


## Prerequisites

This code is developed with

* Python 3.7
* Pytorch 1.8
* Cuda 11.1 

## Citation

```bash
@inproceedings{ distilling-clip-with-dual-guidance-for-learning-discriminative-human-body-shape-representation,
  author = { Feng Liu and Minchul Kim and Zhiyuan Ren and Xiaoming Liu },
  title = { Distilling CLIP with Dual Guidance for Learning Discriminative Human Body Shape Representation },
  booktitle = { In Proceeding of IEEE Computer Vision and Pattern Recognition },
  address = { Seattle, WA },
  month = { June },
  year = { 2024 },
}
```

## Acknowledgments

Here are some great resources we benefit from:

* [Simple-CCReID](https://github.com/guxinqian/Simple-CCReID) for person re-identification.


## License

[MIT License](LICENSE)

## Contact

For questions feel free to post here or drop an email to - liufeng6@msu.edu