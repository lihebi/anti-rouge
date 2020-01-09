dataset_name = "cnn_dailymail"
splits = ['train', 'validation', 'test']
special_characters_to_clean = ['\n', '\t', '\''] # replace such strings with spaces in raw data 
dump_to="'./'+split+method+'.tsv'" # _split_ is an element of _splits_ above 
take_percent =  1  # range from 0 to 100; 100 means all; 0 means none. 
features =['article', 'highlights']
in_memory=False
sent_end = [".","!","?"]  # symbols that represent the end of a sentence 


# parameters for cross pairing
use_cross = False  # True to enable sample generation using cross pairing; False to disable
cross_method = ["cross"] 

# parameters for mutate 
use_mutate = True # True to enable sample generation using mutation; False to disable

mutate_ratios = [10, 30, 50, 70] # percentage
mutate_method = ["add", "delete", "replace"] # comment this line to disable 
