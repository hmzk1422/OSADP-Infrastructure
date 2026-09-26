# researcher-toolkit

The toolkit is the research-side authoring surface for executor features.  The
`live` package deliberately uses a small, serialisable expression graph rather
than trying to translate arbitrary Python bytecode.  This makes the supported
research operations deterministic and lossless when they are evaluated by the
Rust executor.

```python
from live import FeaturePipeline, col

pipeline = FeaturePipeline()
pipeline.AddFeature("spread", col("ask_price") - col("bid_price"))
pipeline.AddFeature("imbalance", pipeline.window("bid_size / ask_size", 20, "mean"))
spec = pipeline.compile()
```

The resulting JSON is the hand-off artifact.  `window` is causal (the current
row and preceding rows only), and `reconstruct` closes only complete bars.
`stresstest` contains checks for parity and boundary behaviour.
