import argparse
import os

# Create an argument parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("-i", "--input_file", help="Input file name", required=True)

parser.add_argument(
    "-v", "--verbose", help="Set logging level to max", action="store_true"
)

# Parse the arguments
args = parser.parse_args()

with open(args.input_file) as f:
    data = f.read().replace("\n", "").replace("    ", "")

import cudnn

if args.verbose:
    os.environ["CUDNN_LOGLEVEL_DBG"] = "3"
else:
    os.environ["CUDNN_LOGLEVEL_DBG"] = "2"

try:
    handle = cudnn.create_handle()

    graph = cudnn.pygraph(handle=handle)

    graph.deserialize(data)

    graph.build([cudnn.heur_mode.A])

    print("Graph built successfully and can be executed.")

except Exception as e:
    print("[cudnn frontend error]")
    print(e)
    print("[cudnn backend error]")
    print(cudnn.get_last_error_string())
