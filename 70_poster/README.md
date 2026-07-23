# [**`70_poster`**](../70_poster/)

This directory contains the scientific poster for the final presentation of the area seminar. The poster summarizes the Oil Slick Detection project, including motivation, methodology, results, conclusion and future work.

By compiling the LaTeX source files in this directory, the poster can be generated as a PDF document:

```bash
latexmk -pdf poster.tex
```

If you are interested in compiling the poster yourself, you should have:

- downloaded the data in [`../10_waterbench_data/`](../10_waterbench_data/)
- run the data analysis in [`../20_data_analysis/`](../20_data_analysis/), and
- executed the experiments in [`../30_experiments/`](../30_experiments).

Based on this, the crucial files for the poster are then saved in [`../50_evaluation/`](../50_evaluation/).
