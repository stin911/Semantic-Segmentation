import logging
from datetime import datetime
import os


def log_funct(training_l, validation_l, epoch):
    now = datetime.now()

    current_time = now.strftime("%H_%M_%S")
    name = "training_"

    logging.basicConfig(filename=os.path.abspath("logging") + "/" + name,
                        format='%(asctime)s %(message)s',
                        filemode='w')

    logging.debug(str(current_time)+" At the epoch" + str(epoch) + " the training loss is " + str(training_l) +
                  " and the validation loss is " + str(validation_l))
    logging.warning(str(current_time) + " At the epoch" + str(epoch) + " the training loss is " + str(training_l) +
                    " and the validation loss is " + str(validation_l))
