import argparse
from synapse_utils import get_syn

def download_file_from_synapse(synapse_id, output_dir):
    syn, temp = get_syn(output_dir)
    syn.get(synapse_id, downloadLocation=output_dir)
    temp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--synapse_id', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)

    args = parser.parse_args()

    download_file_from_synapse(
        args.synapse_id,
        args.output_dir
    )
