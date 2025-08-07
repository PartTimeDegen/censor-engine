# Config Settings Documentation

UPDATED LAST VERSION: 0.1

## 1. Background

## 2. General Settings

### 2.1. File Information

#### 2.1.1. `file_prefix`

    Type: str

    Default: "" TODO

    This setting is used to give censored files a prefix to their name



#### 2.1.2. `file_suffix`

    Type: str

    Default: "" TODO

    This setting is used to give censored files a suffix to their name

#### 2.1.3. `uncensored_folder`

#### 2.1.4. `censored_folder`


### 2.2. `parts_enabled`

---

### 2.3. Image Settings (`image`)

Not Implemented

---

### 2.4. Video Settings (`video`)

Not Implemented

---

### 2.5. Rendering Settings (`rendering`)

`smoothing: bool`

---

### 2.6. Merge Information (`merging`)

#### 2.6.1. `range`

    Type: int

    Not Implemented


#### 2.6.2. `merge_groups`

    Type: list[list[str]]

    blah

---

### 2.7. Part Information (`information`)

### 2.8. `defaults` & *part_settings*

This section refers to settings that appear in both `defaults` and *part_settings*. Defaults refer to parts that aren't explicitly mentioned via a part_setting.

#### 2.8.1. `censors`

Censors are the the styles used for censoring.

The general format for them is:

```
censors:
- function: "top-est censor"
  args:
    arg1: value1
    arg2: value2
    arg3: value3
- function: "middle censor"
  args:
    arg1: value1
    arg2: value2
    arg3: value3
- function: "bottom censor"
  args:
    arg1: value1
    arg2: value2
    arg3: value3
```
It should be noted that the censors should be ordered in the order you want them to be applied.

**Types**

`function: str`

`args: dict[str, str | int | float]`
