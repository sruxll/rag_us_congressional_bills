from marqo import Client
import json
import math
import numpy as np
import copy
import csv  


# convert a CSV file to JSON
def csv_to_json(csv_file_path, json_file_path):
    data = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)


def read_json(filename: str) -> dict:
    """
    Reads a JSON file and returns its content as a dictionary.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file as a dictionary.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def split_big_docs(data, field='text_chunk', char_len=5e4):
    """
    Splits large documents into smaller chunks based on a specified character length.

    Args:
        data (list): A list of dictionaries, each containing a 'content' field or specified field.
        field (str, optional): The field name to check for length. Default is 'content'.
        char_len (float, optional): The maximum character length for each chunk. Default is 5e4.

    Returns:
        list: A list of dictionaries, each containing a chunked version of the original content.
    """
    new_data = []
    for dat in data:
        content = dat[field]
        N = len(content)

        if N >= char_len:
            n_chunks = math.ceil(N / char_len)
            new_content = np.array_split(list(content), n_chunks)

            for _content in new_content:
                new_dat = copy.deepcopy(dat)
                new_dat[field] = ''.join(_content)
                new_data.append(new_dat)
        else:
            new_data.append(dat)
    return new_data

        
def create_vector_index(index_name, data):
    # set up the Client
    mq = Client("http://localhost:8882")
    # delete any existing index if it exists
    try:
        mq.delete_index(index_name)
    except:
        pass
    # create index
    mq.create_index(
        index_name, 
        model='hf/all_datasets_v4_MiniLM-L6'
    )
    # add the subset of data to the index
    mq.index(index_name).add_documents(
        data, 
        client_batch_size=50,
        tensor_fields=["title", "text_chunk"]
    )


def main():
    # process the data as needed and create the index
    csv_file_path = "./parsed_bills_119th.csv"
    dataset_file = "./parsed_bills_119th.json"
    csv_to_json(csv_file_path, dataset_file)
    # get the data
    data = read_json(dataset_file)
    data = split_big_docs(data)
    # take the first 1000 entries of the dataset
    N = 1000 # Number of entries of the dataset
    subset_data = data[:N]
    create_vector_index('text-search-bills', subset_data)

if __name__ == "__main__":
    main()