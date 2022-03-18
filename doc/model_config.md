
## Model Config Convention

### input output name

- input name should follow naming convention: `INPUT_<index>`
- output name should follow naming convention: `OUTPUT_<index>`
- In `INPUT_<index>`, the `<index>` should be in the range `[0, input_size)`
- In `OUTPUT_<index>`, the `<index>` should be in the range `[0, output_size)`

### XRT

You can enable XRT by adding following configuration.

```
parameters {
  key: "xrt"
  value: {
    string_value: "tensorrt"
  }
}
```

### Model Repository Structure

A directory named `model` should be put in the version directory.

Example:

```
.
├── 1
│   └── model
├── client.py
├── config.pbtxt
├── labels.txt
└── model.py
```

#### Model Backend Name

Model backend name must be `oneflow`.

```
name: "identity"
backend: "oneflow"
```
