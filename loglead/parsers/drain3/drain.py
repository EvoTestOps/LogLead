import os.path

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

__all__ = ['DrainTemplateMiner', 'DrainTemplateMinerNoMasking',
           'DrainPersistenceTemplateMiner', 'DrainPersistenceTemplateMinerNoMasking']

current_script_path = os.path.abspath(__file__)
current_script_directory = os.path.dirname(current_script_path)
ini_location = os.path.join(current_script_directory, 'drain3.ini')
tmc = TemplateMinerConfig()
tmc.load(ini_location)
DrainTemplateMiner = TemplateMiner(config=tmc)

persistence = FilePersistence('drain3_state.json')
DrainPersistenceTemplateMiner = TemplateMiner(persistence, config=tmc)


ini_location = os.path.join(current_script_directory, 'drain3_no_masking.ini')
tmc = TemplateMinerConfig()
tmc.load(ini_location)
DrainTemplateMinerNoMasking = TemplateMiner(config=tmc)

persistence_no_masking = FilePersistence('drain3_state_no_masking.json')
DrainPersistenceTemplateMinerNoMasking = TemplateMiner(persistence_no_masking, config=tmc)
