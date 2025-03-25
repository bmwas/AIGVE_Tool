# Latest Research

<style>
    .full-img .nt-card .nt-card-image img {
        height: 300px; /* or whatever height suits your design */
    }
    .full-img .nt-card .nt-card-image {
        min-height: 100%;
        /*aspect-ratio: 16 / 9;  adjust to your needs */
    }
</style>

::cards:: cols=1 class_name="full-img"
- title: "AIGVE-Tool: AI-Generated Video Evaluation Toolkit with Multifaceted Benchmark (March 2025)"
  content: |
    This paper introduces AIGVE-Tool, a unified and modular framework for evaluating AI-generated videos (AIGV) across diverse quality dimensions. As illustrated in Figure 1, AIGVE-Tool features a configuration-driven architecture built upon a novel five-category taxonomy of evaluation metrics, enabling standardized and extensible benchmarking across datasets, models, and modalities. It decouples data handling from metric computation through customizable dataloaders and modular metrics, facilitating seamless integration of new components. Additionally, the authors present AIGVE_Bench, a large-scale benchmark dataset with 2,430 videos and over 21,000 human-annotated scores across nine critical aspects. AIGVE-Tool provides a scalable foundation for advancing fair and comprehensive AIGV evaluation.
  image:
    url: ../assets/img/toolkit_structure.png
    height: 500
  url: "https://arxiv.org/abs/2503.14064"
::/cards::

