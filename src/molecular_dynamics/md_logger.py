from pathlib import Path

class MDLogger:
    def __init__(self, logging_function = None, triggering_function = None, outpath = None, write_to_list = False):
        if logging_function is None:
            self.log = lambda var: str(var.mdsteps)
        else:
            self.log = logging_function
        if triggering_function is None:
            self.trigger = lambda var: var.mdsteps % 1 == 0
        else:
            self.trigger = triggering_function
        self.outfile = outpath
        self.write_to_list = write_to_list
        self.lst = []

    def write_log(self, *args):
        if self.trigger(args[0]):
            log_msg = self.log(args[0])
            if self.outfile is not None:
                self.outfile.write(str(log_msg))
            elif self.write_to_list:
                self.lst.append(log_msg)
            else:
                print(str(log_msg))

    def close_file(self):
        if self.outfile is not None:
            self.outfile.close()