# Gradient Descent Algorithm Implementation

## Install
1) Install anaconda

2) Clone this repo and change directory into to project root (`gradient-descent-algorithm/`)

3) Create a new conda environment
    ```
    conda create -n gd_algorithm python=3.7
    ```
3) Activate the new conda environment
    ```
    activate gd_algorithm
    ```

4) Install pytest in your new conda environment
    ```
    pip install pytest
    ```

## Test
1) Change directory to the project root `gradient-descent-algorithm/` 

2) Make sure you have activated the conda enviornment with `pytest`
    ```
    activate gd_algorithm
    ```
    
3) Run the Pytest cli to execute the test in `gd_solver_test.py`
    ```
    pytest
    ```

    eg. output:
    ```
    ============================= test session starts =============================
    platform win32 -- Python 3.6.6, pytest-3.8.0, py-1.6.0, pluggy-0.7.1
    rootdir: C:\Users\cccarmer\Documents\asu\EEE 511\code\gradient-descent-algorithm, inifile:
    collected 3 items

    gd_solver_test.py ...                                                    [100%]

    ========================== 3 passed in 0.38 seconds ===========================
    ```
