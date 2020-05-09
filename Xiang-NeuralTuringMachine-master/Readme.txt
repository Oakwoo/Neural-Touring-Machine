This code is edited base on the written by Mark(https://github.com/MarkPKCollier/NeuralTuringMachine)
To fit the specific task, I changed some code and added some. You can find them in the comment section at the beginning of each file. There are also some embedded in the code.
To run the program, the packages you need are:
tensorflow
matplotlib
pickle
collections
numpy
scipy
random
and all of .py files in current directory.

To train the NTM model, run "run_tasks.py" and wait for it finished.
After finish the previous step, run "produce_heat_maps.py" and go to "./head_logs/img" to see the testing result.
To see the learning curve, run "draw_curve.py" after you trained the NTM model successfully.

Before training new model, remember to save the current result and delete all *.txt files inside "./head_logs"