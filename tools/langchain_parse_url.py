import warnings
warnings.filterwarnings("ignore")
import pathlib
import sys
import os
import pandas as pd
from typing import List
import multiprocessing
import numpy as np
from tqdm.auto import tqdm
from langchain.document_loaders import UnstructuredURLLoader

ROOT  = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

class Processor(multiprocessing.Process):
    def __init__(self, name, path):
        super().__init__()
        self.name = name
        self.path = path

    def run(self):
        df = pd.read_csv(self.path)

        urls = df["url"].values
        contents = []
        for url in tqdm(urls, bar_format=f"Stock {self.name}" + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining} {postfix}"):
            contents.append(langchain_parse_url(url))
        df["content"] = contents
        df.to_csv(self.path, index=False)

def langchain_parse_url(url):

    urls = [
        url
    ]

    try:
        loader = UnstructuredURLLoader(urls=urls, ssl_verify=False)
        data = loader.load()

        if len(data) <= 0:
            return None

        content = data[0].page_content
    except:
        content = None

    return content

# def main():
#     with open(os.path.join(ROOT, "configs/_stock_list_/dj30.txt")) as op:
#         stocks = [line.strip() for line in op.readlines()]
#     news_paths = os.path.join(ROOT, "datasets/dj30/fmp_news_dj30")
#
#     processes = []
#     for stock in stocks:
#         print(stock)
#         process = Processor(stock, os.path.join(news_paths, f"{stock}.csv"))
#         processes.append(process)
#         process.start()
#
#     for process in processes:
#         process.join()


if __name__ == '__main__':
    from langchain_community.document_loaders import PlaywrightURLLoader

    urls = [
        # "https://www.cnbc.com/2024/01/11/apples-vision-pro-headset-will-be-in-short-supply-at-launch-kuo.html",
        # "https://investorplace.com/2024/01/3-blue-chip-stocks-in-the-tradesmith-green-zone/",
        "https://www.fool.com/investing/2020/01/02/3-stocks-that-turned-1000-into-1-million.aspx"
    ]
    loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
    data = loader.load()[0].page_content
    print(data)

    import requests
    res = requests.get("https://www.marketwatch.com/story/ahead-of-tesla-results-cathie-wood-says-wall-streets-valuing-it-all-wrong-11627298084")
    print(res.text)