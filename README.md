# Nixae
 The code implementation of Nixae.

## Data Format
In a input file, each line represents a *network trace* sample.
Each line consists digital numbers seperated by whitespaces.
These numbers are the byte values in the network trace except for the last one.
And the last number is the class label of the trace.

Sample lines are shown as followings:
```
249 81 1 0 0 1 0 0 0 0 0 1 5 85 83 65 68 70 3 71 79 86 0 0 255 0 1 0 0 41 35 40 0 0 0 0 0 0 0
56 246 0 0 0 1 0 0 0 0 0 0 32 67 75 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 65 0 0 33 0 1 3
```

Use argument `-train_src` to specify your data file.

```
python train_new.py -train_src your_file_path
```

## Notice
This repository is public for review only.
