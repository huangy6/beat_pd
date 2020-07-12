import os
import tempfile
import yaml
import synapseclient as sc

def get_syn(cache_dir):
    username = os.environ['SYNAPSE_USERNAME']
    password = os.environ['SYNAPSE_PASSWORD']
    
    # Trick synapse into actually downloading files where we actually want them - not in ~/.synapseCache
    config = [
        "[cache]",
        f"location = {cache_dir}"
    ]

    # Write the config to a temporary file and pass as configPath= since no other way to override the cache dir?
    temp = tempfile.NamedTemporaryFile(mode='r+')
    temp.write("\n".join(config))
    temp.flush()

    syn = sc.Synapse(configPath=temp.name)
    syn.login(username, password)
    return (syn, temp)