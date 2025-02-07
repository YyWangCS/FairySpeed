This folder contains two PyTorch custom operators `torch.stable_sort` and `torch.stable_sort_opaque`.  It is used to show how `OpaqueType` influences the performance of `torch.sort`. 

```bash
python setup.py install
```

You can find how to use these operators in  `benchmark/profile_stable_sort.py`

> [!NOTE]
>
> The code is based on https://github.com/pytorch/extension-cpp.git. 

