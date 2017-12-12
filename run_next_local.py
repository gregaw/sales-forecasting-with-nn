import os

from trainer import task

epochs = 50
eval_epoch_freq = 10


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
--train-files data/train1.csv 
--eval-files data/eval1.csv 
--job-dir output/{} 
--num-epochs {}
--checkpoint-epochs {}
--eval-frequency {}
""".format(run_name, epochs, eval_epoch_freq, eval_epoch_freq)

parsed_args = task.parse_args(args.replace('\n', ' ').replace('  ', ' ').split(" "))

task.dispatch(**parsed_args.__dict__)
