from pathlib import Path

class MDLogger:
    def __init__(self, logging_function = None, triggering_function = None, outpath= None):
        if logging_function is None:
            self.log = lambda var: str(var.mdsteps)
        else:
            self.log = logging_function
        if triggering_function is None:
            self.trigger = lambda var: var.mdsteps % 1 == 0
        else:
            self.trigger = triggering_function
        self.outpath = outpath
        self.outfile = None


    def initialise_file(self):
        if self.outfile is not None and not Path(self.outfile).is_file():
            self.outfile = open(self.outfile, "w")

    def write_log(self, *args):
        if self.trigger(args[0]):
            log_msg = self.log(args[0])
            if self.outfile is None:
                print(log_msg)
            else:
                self.outfile.write(log_msg)

    def close_file(self):
        if self.outfile is not None:
            self.outfile.close()



