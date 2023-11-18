#
#Separate demo files
import loglead.loader as load, loglead.enhancer as er, anomaly_detection as ad


gelf_processor = load.GELFLoader(filename="gelf.log")
df_gelf = gelf_processor.execute()

#Quick stuff to test:

enhancer_gelf = er.EventLogEnhancer(df_gelf)
df_gelf = enhancer_gelf.words()
df_gelf = enhancer_gelf.alphanumerics()
df_gelf = enhancer_gelf.trigrams()
df_gelf = enhancer_gelf.parse_drain()

event_anomaly_detection = ad.EventAnomalyDetection(df_gelf)
df_gelf = event_anomaly_detection.compute_ano_score("e_words", 100)
df_gelf = event_anomaly_detection.compute_ano_score("e_alphanumerics", 100)
df_gelf = event_anomaly_detection.compute_ano_score("e_trigrams", 100)