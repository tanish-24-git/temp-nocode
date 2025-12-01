```
{
  "global": {
    "missing_strategy": "mean",    // fallback for numeric cols if not set
    "scaling": true,               // whether numeric scaling is default
    "encoding": "onehot"           // default encoding for categorical
  },
  "columns": {
    "age": {
      "missing_strategy": "median",
      "scaling": true,
      "encoding": "passthrough",   // numeric -> passthrough (no change)
      "action": "keep"             // values: "keep", "drop", "custom"
    },
    "country": {
      "missing_strategy": "mode",
      "encoding": "onehot",
      "action": "keep"
    },
    "id": {
      "action": "drop"
    },
    "salary": {
      "scaling": false,
      "missing_strategy": "mean",
      "action": "keep"
    }
  },
  "custom_transformers": {
    "salary_log": {
      "type": "lambda",
      "params": { "expr": "np.log1p" },
      "applies_to": ["salary"]
    }
  }
}
```