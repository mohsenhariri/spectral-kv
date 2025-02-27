import argparse

parser = argparse.ArgumentParser(description="URL of file or folder")

parser.add_argument("-id", type=str, default=None, metavar="N", help="URL")
id = parser.parse_args().id

print()
