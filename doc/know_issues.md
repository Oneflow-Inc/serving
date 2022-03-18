
## Known Issues

### llvm: Option already exists

Oneflow backend conflits with tensorflow1 due to some mysterious reason. It is recommended not to use oneflow and tensorflow1 together.

```
.../llvm/include/llvm/Support/CommandLine.h:858: void llvm::cl::parser<DataType>::addLiteralOption
(llvm::StringRef, const DT&, llvm::StringRef) [with DT = llvm::FunctionPass* (*)(); DataType = 
llvm::FunctionPass* (*)()]: Assertion `findOption(Name) == Values.size() && "Option already 
exists!"' failed.
```

### Multiple model instance executaion

The current version of oneflow does not support concurrent execution of multiple model instances. You can launch multiple container with k8s to bypass this limitation.

