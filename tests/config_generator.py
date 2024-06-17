import tkinter as tk
from tkinter import ttk, messagebox
import yaml
import os
import runpy
import sys

"""
This UI based tool is primarily meant for generating more varied configuration files faster. It can also be more approachable to some users.

Profilence cannot be downloaded because it is not a public dataset. ADFA and AWSCTD should not be enhanced because they are already parsed events.

The buttons to run the scripts in the UI always fetches the .yml file configured in the text box. Note that it might be different than the selection in the UI.
"""

# Static values and defaults
datasets = [
    {
        'name': 'bgl',
        'url': 'https://zenodo.org/records/8196385/files/BGL.zip?download=1',
        'log_file': 'BGL.log',
        'download': True,
        'load': True,
        'enhance': True,
        'anomaly_detection': True,
        'expected_length': 4747963,
        'reduction_fraction': 0.02
    },
    {
        'name': 'hadoop',
        'url': 'https://zenodo.org/records/8196385/files/Hadoop.zip?download=1',
        'filename_pattern': '*.log',
        'labels_file': 'abnormal_label.txt',
        'download': True,
        'load': True,
        'enhance': True,
        'anomaly_detection': True,
        'expected_length': 180897,
        'reduction_fraction': 0.56
    },
    {
        'name': 'hdfs',
        'url': 'https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1',
        'log_file': 'HDFS.log',
        'labels_file': 'preprocessed/anomaly_label.csv',
        'download': True,
        'load': True,
        'enhance': True,
        'anomaly_detection': True,
        'expected_length': 11175629,
        'reduction_fraction': 0.01
    },
    {
        'name': 'liberty',
        'url': 'http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/liberty2.gz',
        'log_file': 'liberty2.log',
        'download': False,
        'load': False,
        'enhance': False,
        'anomaly_detection': False,
        'expected_length': 0,
        'reduction_fraction': 0.0005
    },
    {
        'name': 'spirit',
        'url': 'http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/spirit2.gz',
        'log_file': 'spirit2.log',
        'download': False,
        'load': False,
        'enhance': False,
        'anomaly_detection': False,
        'expected_length': 0,
        'reduction_fraction': 0.0005
    },
    {
        'name': 'thunderbird',
        'url': 'http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/tbird2.gz',
        'log_file': 'tbird2.log',
        'download': False,
        'load': False,
        'enhance': False,
        'anomaly_detection': False,
        'expected_length': 211212192,
        'reduction_fraction': 0.0005
    },
    {
        'name': 'nezha',
        'urls': [
            'https://github.com/IntelligentDDS/Nezha/tree/main/construct_data/',
            'https://github.com/IntelligentDDS/Nezha/tree/main/rca_data/'
        ],
        'systems': [
            'TrainTicket',
            'WebShop'
        ],
        'download': False,
        'load': False,
        'enhance': False,
        'anomaly_detection': False,
        'train_ticket': {
            'expected_length': 272270,
            'reduction_fraction': 0.33
        },
        'web_shop': {
            'expected_length': 3958203,
            'reduction_fraction': 0.025
        }
    },
    {
        'name': 'profilence',
        'log_file': '*.txt',
        'download': False,
        'load': False,
        'enhance': False,
        'anomaly_detection': False,
        'expected_length': 5203599,
        'reduction_fraction': 0.02
    },
    {
        'name': 'adfa',
        'url': 'https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset/raw/master/ADFA-LD.zip',
        'folder': 'ADFA-LD',
        'download': False,
        'load': False,
        'enhance': False,  # Always skip enhancement
        'anomaly_detection': False,
        'expected_length': 2747550,
        'reduction_fraction': 0.0364
    },
    {
        'name': 'awsctd',
        'url': 'https://github.com/DjPasco/AWSCTD/raw/master/CSV.7z',
        'folder': 'AWSCTD',
        'download': False,
        'load': False,
        'enhance': False,  # Always skip enhancement
        'anomaly_detection': False,
        'expected_length': 174847810,
        'reduction_fraction': 0.0057
    }
]


class ConfigGenerator(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Test Configuration Generator")
        self.geometry("900x400")

        self.dataset_vars = []
        self.reduction_var = tk.IntVar(value=100)  # Default reduction target is 100k lines
        self.create_widgets()

    def create_widgets(self):
        self.frame = ttk.Frame(self)
        self.frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(self.frame)
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Track the current column and row
        current_col = 0
        current_row = 0
        max_cols = 5

        for dataset in datasets:
            dataset_frame = ttk.LabelFrame(self.scrollable_frame, text=dataset['name'])
            dataset_frame.grid(row=current_row, column=current_col, padx=10, pady=5, sticky="nsew")

            var_download = tk.BooleanVar(value=dataset['download'])
            var_load = tk.BooleanVar(value=dataset['load'])
            var_enhance = tk.BooleanVar(value=dataset['enhance'])
            var_anomaly_detection = tk.BooleanVar(value=dataset['anomaly_detection'])

            if dataset['name'] == 'nezha':
                self.dataset_vars.append({
                    'name': dataset['name'],
                    'download': var_download,
                    'load': var_load,
                    'enhance': var_enhance,
                    'anomaly_detection': var_anomaly_detection,
                    'train_ticket': dataset['train_ticket']['expected_length'],
                    'web_shop': dataset['web_shop']['expected_length']
                })
            else:
                self.dataset_vars.append({
                    'name': dataset['name'],
                    'download': var_download,
                    'load': var_load,
                    'enhance': var_enhance,
                    'anomaly_detection': var_anomaly_detection,
                    'expected_length': dataset['expected_length']
                })

            ttk.Checkbutton(dataset_frame, text="Download", variable=var_download, state='disabled' if dataset['name'] == 'profilence' else 'normal').pack(anchor='w')
            ttk.Checkbutton(dataset_frame, text="Load", variable=var_load).pack(anchor='w')
            ttk.Checkbutton(dataset_frame, text="Enhance", variable=var_enhance, state='disabled' if dataset['name'] in ['adfa', 'awsctd'] else 'normal').pack(anchor='w')
            ttk.Checkbutton(dataset_frame, text="Anomaly Detection", variable=var_anomaly_detection).pack(anchor='w')

            # Move to the next column, and wrap to the next row if needed
            current_col += 1
            if current_col >= max_cols:
                current_col = 0
                current_row += 1

        # Create a frame for reduction target and save location
        options_frame = ttk.Frame(self)
        options_frame.pack(pady=10, fill=tk.X)

        ttk.Label(options_frame, text="Reduction Target (applied in loading):").grid(row=0, column=0, padx=10)
        reductions = [10, 50, 100, 150]
        reduction_frame = ttk.Frame(options_frame)
        reduction_frame.grid(row=0, column=1, padx=5)
        for i, reduction in enumerate(reductions):
            ttk.Radiobutton(reduction_frame, text=f"{reduction}k", variable=self.reduction_var, value=reduction).pack(side=tk.LEFT, padx=2)

        self.data_location_label = ttk.Label(options_frame, text="Data location:")
        self.data_location_label.grid(row=1, column=0, padx=10)

        default_data_path = os.path.expanduser("~/Datasets")
        self.data_location_entry = ttk.Entry(options_frame, width=60)
        self.data_location_entry.grid(row=1, column=1, padx=10)
        self.data_location_entry.insert(0, default_data_path)  # Default data location

        self.save_location_label = ttk.Label(options_frame, text="Config save/run location:")
        self.save_location_label.grid(row=2, column=0, padx=10)

        default_save_path = os.path.abspath("datasets_generated.yml")
        self.save_location_entry = ttk.Entry(options_frame, width=60)
        self.save_location_entry.grid(row=2, column=1, padx=10)
        self.save_location_entry.insert(0, default_save_path)  # Default save location

        self.save_button = ttk.Button(options_frame, text="Save Configuration", command=self.save_config)
        self.save_button.grid(row=2, column=2, padx=10)

        # Create a frame for the run buttons
        run_buttons_frame = ttk.Frame(self)
        run_buttons_frame.pack(pady=10, fill=tk.X)

        ttk.Label(run_buttons_frame, text="Run scripts (see console):").grid(row=0, column=0, padx=10)

        ttk.Button(run_buttons_frame, text="Run Download Data", command=lambda: self.run_script('download_data.py')).grid(row=0, column=1, padx=10)
        ttk.Button(run_buttons_frame, text="Run Loaders", command=lambda: self.run_script('loaders.py')).grid(row=0, column=2, padx=10)
        ttk.Button(run_buttons_frame, text="Run Enhancers", command=lambda: self.run_script('enhancers.py')).grid(row=0, column=3, padx=10)
        ttk.Button(run_buttons_frame, text="Run Anomaly Detectors", command=lambda: self.run_script('anomaly_detectors.py')).grid(row=0, column=4, padx=10)

    def save_config(self):
        config = {'root_folder': self.data_location_entry.get(), 'datasets': []}
        reduction_target = self.reduction_var.get()

        for dataset, vars in zip(datasets, self.dataset_vars):
            if dataset['name'] == 'nezha':
                reduction_fraction_tt = reduction_target * 1000 / vars['train_ticket']
                reduction_fraction_ws = reduction_target * 1000 / vars['web_shop']
                dataset_config = {
                    'name': dataset['name'],
                    'urls': dataset['urls'],
                    'systems': dataset['systems'],
                    'download': vars['download'].get(),
                    'load': vars['load'].get(),
                    'enhance': vars['enhance'].get(),
                    'anomaly_detection': vars['anomaly_detection'].get(),
                    'train_ticket': {
                        'expected_length': vars['train_ticket'],
                        'reduction_fraction': reduction_fraction_tt
                    },
                    'web_shop': {
                        'expected_length': vars['web_shop'],
                        'reduction_fraction': reduction_fraction_ws
                    }
                }
            else:
                reduction_fraction = reduction_target * 1000 / vars['expected_length'] if vars['expected_length'] else 0
                dataset_config = {
                    'name': dataset['name'],
                    'url': dataset.get('url', ''),
                    'log_file': dataset.get('log_file', ''),
                    'filename_pattern': dataset.get('filename_pattern', ''),
                    'labels_file': dataset.get('labels_file', ''),
                    'download': vars['download'].get(),
                    'load': vars['load'].get(),
                    'enhance': vars['enhance'].get(),
                    'anomaly_detection': vars['anomaly_detection'].get(),
                    'expected_length': vars['expected_length'],
                    'reduction_fraction': reduction_fraction
                }

            config['datasets'].append(dataset_config)

        file_path = self.save_location_entry.get()
        if file_path:
            with open(file_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)
            messagebox.showinfo("Success", f"Configuration saved to {file_path}")

    def run_script(self, script_name):
        file_path = self.save_location_entry.get()
        if os.path.exists(file_path):
            print(f"Using configuration file: {file_path}")
            original_argv = sys.argv
            sys.argv = [script_name, '--config', file_path]
            try:
                runpy.run_path(script_name, run_name='__main__')
            finally:
                sys.argv = original_argv
            print("___________________________________________________")
        else:
            messagebox.showerror("Error", f"Configuration file not found: {file_path}")

if __name__ == "__main__":
    app = ConfigGenerator()
    app.mainloop()
