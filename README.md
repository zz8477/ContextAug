### This is the online repository of a test case augmentation technique.
## Task Definition

Model-domain Synthesized Failing Test Augmentation for Fault Localization with Smote and Slice.

## Dataset

The datasets we used are from [Defects4J](http://defects4j.org),[ManyBugs](http://repairbenchmarks.cs.umass.edu/ManyBugs/),[SIR](http://sir.unl.edu/portal/index.php).

This is a replication package for the task with nanoxml_v2
source code:StdXMLParser.java
buggy line:363 

## Data Format

1. componentinfo.txt is stored in txt format. The first row is the number of total executed statements. The second row is the line number of each statement.

2. covMatrix.txt is stored in txt format. It is the model-domain test cases. Each row is a test case, the element 1 or 0 is the coverage information of statements, in which 1 denotes the corresponding statement is executed by the test case, and 0 otherwise.

3. error.txt is stored in txt format. It is the test cases' results, each line represents one result. 1 means a failed test case and 0 denotes a successful test case.

4. sliceResult.txt is stored in txt format. It is the slice result. The 1st row is the total number of statements included in the slice while the 2rd row contains the line number of statements.

5. result/newMatrix.txt is stored in txt format. It is result of the generated model-domain test cases, each row represents one generated model-domain test case. the elements are the value of statements.

6. result/newError.txt is stored in txt format. It is the label of the generated model-domain test cases, each row represents one label for the corresponding generated model-domain test case, the value is 1 which denotes that it is a label of model-domain synthesized failing test.

7. result/cos_result.txt is stored in txt format. It is the Euclidean distance of each input-domain failing test case with other ones.

8. result/result.txt is stored in txt format. It includes all the input-domain failing test cases as well as their positions.

## Usage
You can get generated model-domain test cases' results using the following command.

```
python testcase_aug.py dev
```
Generated model-domain test cases' results are in result/newMatrix.txt and result/newError.txt

## Dependency
- gcc >= 3.3.1
- python version: python3.7.6
- pip install torch
- pip install numpy
## Evaluator
You can get the evaluated results with synthesized failing test augmentation using the following command.
```
cd SFL_compute_augmentation
./run_aug.sh
```
Fault localization results are in result.txt, the 1st element from the left is the method name, the 2rd is the value of Rank while the 4th is the value of EXAM.
You can get the original evaluated results using the following command.
```
cd SFL_compute_original
./run_orig.sh
```
Fault localization results are in result.txt, the 1st element from the left is the method name, the 2rd is the value of Rank while the 4th is the value of EXAM.

