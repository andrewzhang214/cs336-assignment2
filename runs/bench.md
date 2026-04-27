| model_size   |   batch_size |   vocab_size |   context_length | mode     |   num_warmup_steps |   num_measure_steps |   mean_ms |   std_ms | device   |
|:-------------|-------------:|-------------:|-----------------:|:---------|-------------------:|--------------------:|----------:|---------:|:---------|
| small        |            4 |        10000 |              128 | forward  |                  5 |                  10 |     0.133 |    0.001 | mps      |
| small        |            4 |        10000 |              128 | backward |                  5 |                  10 |     0.085 |    0.006 | mps      |
| medium       |            4 |        10000 |              128 | forward  |                  5 |                  10 |     0.854 |    0.382 | mps      |
| medium       |            4 |        10000 |              128 | backward |                  5 |                  10 |     0.833 |    0.536 | mps      |