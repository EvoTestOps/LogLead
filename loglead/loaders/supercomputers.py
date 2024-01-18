from loglead.loaders.base import BaseLoader
import polars as pl
# Processor for the Thunderbird, Spirit and Liberty log files
class ThuSpiLibLoader(BaseLoader):
    def __init__(self, filename, df=None, df_seq=None, split_component=True):
        self.split_component = split_component
        super().__init__(filename, df, df_seq)

        
    def load(self):
        self.df = pl.read_csv(self.filename, has_header=False, infer_schema_length=0, 
                              separator=self._csv_separator, ignore_errors=True) #There is one UTF error in the file
    
    
    def preprocess(self):
        if self.split_component:
            self._split_and_unnest(["label", "timestamp", "date", "userid", "month", 
                                    "day", "time", "location", "component_pid", "m_message"])
            self._split_component_and_pid()
        else:
            self._split_and_unnest(["label", "timestamp", "date", "userid", "month", 
                                    "day", "time", "location", "m_message"])
        #parse datatime
        self.df = self.df.with_columns(m_timestamp = pl.from_epoch(pl.col("timestamp")))
        #Label contains multiple anomaly cases. Convert to binary
        self.df = self.df.with_columns(normal = pl.col("label").str.starts_with("-"))

    #Reason for extra processing. We want so separte pid from component and in the log file they are embedded
    #Data description
    #https://github.com/logpai/loghub/blob/master/Thunderbird/Thunderbird_2k.log_structured.csv     
    def _split_component_and_pid(self):
        component_and_pid = self.df.select(pl.col("component_pid")).to_series().str.splitn("[", n=2)
        component_and_pid = component_and_pid.struct.rename_fields(["component", "pid"])
        component_and_pid = component_and_pid.alias("fields")
        component_and_pid = component_and_pid.to_frame()
        component_and_pid = component_and_pid.unnest("fields")
        component_and_pid = component_and_pid.with_columns(pl.col("component").str.rstrip(":"))
        component_and_pid = component_and_pid.with_columns(pl.col("pid").str.rstrip("]:"))
        self.df= pl.concat([self.df, component_and_pid], how="horizontal")  
        self.df = self.df.drop("component_pid")
        self.df = self.df.select(["label", "timestamp", "date", "userid", "month", 
                                "day", "time", "location", "component","pid", "m_message"])

# Processor for the BGL log file
# At the moment there are 34470 null messages that are not handled by the loader
class BGLLoader(BaseLoader):
    def load(self):
        self.df = pl.read_csv(self.filename, has_header=False, infer_schema_length=0, 
                              separator=self._csv_separator, ignore_errors=True)
    def preprocess(self):
        self._split_and_unnest(["label", "timestamp", "date", "node", "time", 
                            "noderepeat", "type", "component", "level", "m_message"])
        self.df = self.df.with_columns(normal = pl.col("label").str.starts_with("-")) #Same format as Tb
        #parse datatime
        self.df = self.df.with_columns(m_timestamp = pl.from_epoch(pl.col("timestamp")))
      