
# =========================================================================
# Copyright (C) 2016-2023 LOGPAI (https://github.com/logpai).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import pandas as pd
import regex as re
import datetime
#Routine loads data from logfile and creates a dataframe. 
#Routine taken from LogPai/Logpaser/Drain project
#https://github.com/logpai/logparser/blob/main/logparser/Drain/Drain.py#L327C1-L344C21
#Following modification were made.
#1) Converted to normal function instead of class member function
#2) Added a counter that reports progress every 1,000,000 rows
#3) Added errors=ignore as there is a non-utf character in TB.
#4) Moved generate_logformat_regex inside this function 
def log_to_dataframe(log_file, log_format):
    """Function to transform log file to dataframe"""
    headers, regex = generate_logformat_regex(log_format)
    log_messages = []
    linecount = 0
#    with open(log_file, "r") as fin:
    with open(log_file, "r", errors="replace") as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
                if linecount % 1000000 == 0:
                    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}", flush=True)
            except Exception as e:
                continue
                #print("[Warning] Skip line: " + line)
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, "LineId", None)
    logdf["LineId"] = [i + 1 for i in range(linecount)]
    print("Total lines: ", len(logdf))
    return logdf

# Routine from https://github.com/logpai/logparser/blob/main/logparser/Drain/Drain.py#L346
#Changes
#1) Changed to normal function from class member function
def generate_logformat_regex(logformat):
    """Function to generate regular expression to split log messages"""
    headers = []
    splitters = re.split(r"(<[^<>]+>)", logformat)
    regex = ""
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(" +", "\\\s+", splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip("<").strip(">")
            regex += "(?P<%s>.*?)" % header
            headers.append(header)
    regex = re.compile("^" + regex + "$")
    return headers, regex        

