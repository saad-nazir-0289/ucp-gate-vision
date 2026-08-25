# sample_data/

Gitignored. Put local test video files here, e.g. `sample_data/gate1_sample.mp4`, then run:

```bash
python run_pipeline.py --video sample_data/gate1_sample.mp4
```

See the "Testing" section of [`../README.md`](../README.md) for where to
source a sample video and how to evaluate accuracy.

## Guidlines for ground_truth.csv

Each undefined value means, number plate wasn't readable from humen eye.
rainy data isn't available
Dataset is sent on whatsapp group (Vision Project) file_name data_set.zip. It contains these file
    dataset_clear_01.mp4
    dataset_dusk_01.mp4
    dataset_night_01.mp4
    dataset_multiple_vehicles_01.mp4
    dataset_multiple_vehicles_02.mp4