# Task 3
This is the source code for task 3.

## Installing Dependencies:
```pip install -r requirements.txt ```

## Running the code:
If you have unzipped this from submission, the test files are already included in the relative paths.
The entrypoint is in main.py:

```python3 main.py```

If for some reason the test files are not included, they should be placed in the following relative paths:

```./Task3AdditionalTestDataset/```

and 


```./TestWithoutRotations/```

and 

```./Training/```

## Configuring the code
There is only really one setting to change: the SHOW_WINDOWS flag. 
By default this is True, causing a windows to be shown indicating the matches found.
A key press is needed to move onto the next image.
Setting this flag to False (line 10 of main.py) will disable this behaviour and just output the evaluations.