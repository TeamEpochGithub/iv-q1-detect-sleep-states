# Base class for preprocessing
import os
import pandas as pd
import gc


class PP:
    def __init__(self):
        self.use_pandas = True

    def preprocess(self, data):
        raise NotImplementedError

    def run(self, data, config):
        processed = data
        steps, step_names = config.get_pp_steps()
        for i in range(len(step_names), -1, -1):
            path = config.get_pp_out() + '/' + '_'.join(step_names[:i]) + '.parquet'
            # check if the final result of the preprocessing exists
            if os.path.exists(path):
                print('Found existing file at:', path)
                print('Reading from:', path)
                processed = pd.read_parquet(path)
                print('Data read from:', path)
                break
            else:
                if i == 0:
                    print('No files found, reading from:', config.get_pp_in())
                else:
                    print('File not found at:', path)
                # find the latest version of the preprocessing
                # inside this loop
                continue
        # if no files were found, i=0, read the unprocessed data here
        if i == 0:
            print('No files found, reading from:', config.get_pp_in())
            processed = pd.read_parquet(config.get_pp_in() + '/train_series.parquet')
            print('Data read from:', config.get_pp_in())
        # now using i run the preprocessing steps that were not applied
        for j, step in enumerate(step_names[i:]):
            path = config.get_pp_out() + '/' + '_'.join(step_names[:i+j+1]) + '.parquet'
            # step is the string name of the step to apply
            step = steps[i+j]
            print('Applying preprocessing step:', step_names[i+j])
            processed = step.preprocess(processed)
            gc.collect()
            # save the result
            print('Preprocessing was applied')
            print('Saving to:', path)
            processed.to_parquet(path)
            print('Saved to: ', path)
        return processed
