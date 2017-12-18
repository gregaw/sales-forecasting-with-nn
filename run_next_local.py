import os

from trainer import task

epochs = 300
eval_epoch_freq = 25


# ======================================================

def next_run_number(prefix='mac_'):
    """
    Calculates the next run number as the largest+1 from the list of prefix+number in the output folder

    :param prefix: folder prefix in output
    :return: next run number
    """
    runs = os.listdir('output')
    runs = filter(lambda x: x.startswith(prefix), runs)
    runs = sorted(runs, reverse=True)
    next_run = prefix + str(int(runs[0][len(prefix):]) + 1).zfill(6) if len(runs) else prefix+"0".zfill(6)
    return next_run


run_name = next_run_number()

args = """
--train-files data/train2.csv data/train3.csv data/train5.csv data/train6.csv data/train10.csv data/train14.csv data/train17.csv data/train22.csv data/train27.csv data/train4.csv data/train21.csv data/train63.csv data/train69.csv data/train70.csv data/train99.csv data/train141.csv data/train150.csv data/train203.csv data/train232.csv
--eval-files data/eval28.csv data/eval30.csv data/eval255.csv data/eval293.csv
--job-dir output/{} 
--num-epochs {}
--checkpoint-epochs {}
--eval-frequency {}
""".format(run_name, epochs, eval_epoch_freq, eval_epoch_freq)

parsed_args = task.parse_args(args.replace('\n', ' ').replace('  ', ' ').split(" "))

task.dispatch(**parsed_args.__dict__)


#sample params
#--train-files data/train1.csv
#--eval-files data/eval1.csv

# sample set of data with 10 different stores of type a + 10 different stores of type c and test on 4 different stores of type 2x a + 2x c
#--train-files data/train2.csv data/train3.csv data/train5.csv data/train6.csv data/train10.csv data/train14.csv data/train17.csv data/train22.csv data/train27.csv data/train4.csv data/train21.csv data/train63.csv data/train69.csv data/train70.csv data/train99.csv data/train141.csv data/train150.csv data/train203.csv data/train232.csv
#--eval-files data/eval28.csv data/eval30.csv data/eval255.csv data/eval293.csv