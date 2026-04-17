# References

<!-- This file collects the primary papers and official resources for the methods, metrics, models, and benchmarks mentioned in the README.

## Benchmarks and project context

1. **Towards True Detail Restoration for Super-Resolution: A Benchmark and a Quality Metric**  
   Eugene Lyapustin, Anastasia Kirillova, Viacheslav Meshchaninov, Evgeney Zimin, Nikolai Karetin, Dmitriy Vatolin. 2022.  
   arXiv: https://arxiv.org/abs/2203.08923

2. **Super-Resolution Quality Metrics Benchmark**  
   MSU Video Processing benchmark page.  
   https://videoprocessing.ai/benchmarks/super-resolution-metrics.html

3. **Super-Resolution Quality Metrics Benchmark Methodology**  
   MSU Video Processing methodology page.  
   https://videoprocessing.ai/benchmarks/super-resolution-metrics-methodology.html -->

## No-Reference (NR) IQA metrics

1. **Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels**  
   Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtao Zhai, Weisi Lin. 2023.  
   arXiv: https://arxiv.org/abs/2312.17090

2. **MUSIQ: Multi-scale Image Quality Transformer**  
   Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, Feng Yang. 2021.  
   arXiv: https://arxiv.org/abs/2108.05997

3. **ARNIQA: Learning Distortion Manifold for Image Quality Assessment**  
   Lorenzo Agnolucci, Leonardo Galteri, Marco Bertini, Alberto Del Bimbo. 2023.  
   arXiv: https://arxiv.org/abs/2310.14918

4. **UNIQUE: Unsupervised Image Quality Estimation**  
   Dogancan Temel, Mohan Prabhushankar, Ghassan AlRegib. 2018.  
   arXiv: https://arxiv.org/abs/1810.06631

5. **From Patches to Pictures (PaQ-2-PiQ): Mapping the Perceptual Space of Picture Quality**  
   Zhenqiang Ying, Haoran Niu, Praful Gupta, Dhruv Mahajan, Deepti Ghadiyaram, Alan Bovik. 2020.  
   arXiv: https://arxiv.org/abs/1912.10088  
   <!-- CVPR OpenAccess PDF: https://openaccess.thecvf.com/content_CVPR_2020/papers/Ying_From_Patches_to_Pictures_PaQ-2-PiQ_Mapping_the_Perceptual_Space_of_CVPR_2020_paper.pdf -->

## Full-Reference (FR) IQA metrics

6. **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric**  
   Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang. 2018.  
   arXiv: https://arxiv.org/abs/1801.03924  
   <!-- Project page: https://richzhang.github.io/PerceptualSimilarity/ -->

7. **Shift-tolerant Perceptual Similarity Metric**  
    Aseem Ghildyal, Feng Liu. 2022.  
    arXiv: https://arxiv.org/abs/2207.13686

8. **PieAPP: Perceptual Image-Error Assessment through Pairwise Preference**  
    Ekta Prashnani, Hong Cai, Yasamin Mostofi, Pradeep Sen. 2018.  
    arXiv: https://arxiv.org/abs/1806.02067  
    <!-- CVPR OpenAccess PDF: https://openaccess.thecvf.com/content_cvpr_2018/papers/Prashnani_PieAPP_Perceptual_Image-Error_CVPR_2018_paper.pdf -->

9. **Attentions Help CNNs See Better: Attention-based Hybrid Image Quality Assessment Network**  
    Shanshan Lao, Yuan Gong, Shuwei Shi, Sidi Yang, Tianhe Wu, Jiahao Wang, Weihao Xia, Yujiu Yang. 2022.  
    arXiv: https://arxiv.org/abs/2204.10485  
    <!-- Code: https://github.com/IIGROUP/AHIQ -->

10. **Image Quality Assessment: From Error Visibility to Structural Similarity**  
    Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, Eero P. Simoncelli. 2004.  
    Official page: https://ece.uwaterloo.ca/~z70wang/publications/ssim.html  
    <!-- PDF: https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf -->

<!-- 14. **PSNR / MSE-based fidelity measures**  
    PSNR is a standard analytical metric derived from mean squared error rather than a single canonical method paper.  
    A useful historical reference in the IQA literature is:  
    **Image Quality Assessment: From Error Visibility to Structural Similarity**  
    Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, Eero P. Simoncelli. 2004.  
    https://ece.uwaterloo.ca/~z70wang/publications/ssim.html -->

## Pretrained encoder backbones

11. **Very Deep Convolutional Networks for Large-Scale Image Recognition**  
    Karen Simonyan, Andrew Zisserman. 2015.  
    arXiv: https://arxiv.org/abs/1409.1556  
    <!-- VGG page: https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/ -->

12. **Deep Residual Learning for Image Recognition**  
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. 2015.  
    arXiv: https://arxiv.org/abs/1512.03385  
    <!-- CVPR OpenAccess PDF: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf -->

13. **Sigmoid Loss for Language Image Pre-Training**  
    Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer. 2023.  
    arXiv: https://arxiv.org/abs/2303.15343

## Reference / quasi-GT upscaling methods

14. **Swift Parameter-free Attention Network for Efficient Super-Resolution**  
    Cheng Wan, Hongyuan Yu, Zhiqi Li, Yihang Chen, Yajun Zou, Yuqing Liu, Xuanwu Yin, Kunlong Zuo. 2023.  
    arXiv: https://arxiv.org/abs/2311.12770  
    <!-- Code: https://github.com/hongyuanyu/SPAN -->

15. **Residual Local Feature Network for Efficient Super-Resolution**  
    Fangyuan Kong, Mingxi Li, Songwei Liu, Ding Liu, Jingwen He, Yang Bai, Fangmin Chen, Lean Fu. 2022.  
    arXiv: https://arxiv.org/abs/2205.07514  
    <!-- CVPRW PDF: https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Kong_Residual_Local_Feature_Network_for_Efficient_Super-Resolution_CVPRW_2022_paper.pdf -->
<!-- 
## Tooling

20. **IQA-PyTorch / PyIQA**  
    Official repository: https://github.com/chaofengc/IQA-PyTorch  
    Documentation: https://iqa-pytorch.readthedocs.io/

## Notes

- `LPIPS-VGG` in the README is covered by the LPIPS paper above.
- `STLPIPS-VGG` is mapped here to the shift-tolerant LPIPS line of work via the **Shift-tolerant Perceptual Similarity Metric** paper.
- Bicubic interpolation is a standard resampling method and is therefore not tied to a single modern paper in the same way as the learned SR methods above. -->