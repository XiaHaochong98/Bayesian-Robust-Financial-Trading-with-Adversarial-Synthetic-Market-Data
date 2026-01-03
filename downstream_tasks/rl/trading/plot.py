import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import json5
import pyecharts.options as opts
from pyecharts.charts import Line, Grid
from pyecharts.render import make_snapshot
# from snapshot_selenium import snapshot as driver
from snapshot_phantomjs import snapshot as driver

os.environ["QT_QPA_PLATFORM"] = "offscreen"

ROOT = str(Path(__file__).resolve().parents[4])
CURRENT = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CURRENT)


def parse_args():
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument("--dir_path", default=os.path.join("downstream_tasks/rl/trading/workdir/exp/trading/"),
                        help="json directory path")
    parser.add_argument("--stock", type=str, default="AAPL")
    parser.add_argument("--algo", type=str, default="ppo")
    args = parser.parse_args()
    return args


def plot_graph(selected_stock, dir_path, selected_algorithm, tag, stage, now_date=None):
    input_path = os.path.join(dir_path, selected_stock, selected_algorithm, tag, f'{stage}_records.json')
    output_path = os.path.join(CURRENT, "workdir/fig", f"{selected_algorithm}_{selected_stock}_{tag}_{stage}.png")

    print("attemp to open: ", input_path)
    try:
        data = json5.load(open(input_path))
    except Exception as e:
        print(f'Records file {input_path} does not exist!')
        return

    dates = [datetime.strptime(date[0], '%Y-%m-%d') for date in data['timestamp']][:-1]

    closing_prices = [item_list[0] for item_list in data['price']][:-1]
    returns = [item_list[0] for item_list in data['total_profit']][1:]
    actions = [item_list[0] for item_list in data['action']][1:]

    min_y = min(closing_prices)
    max_y = max(closing_prices)
    delta = max_y - min_y
    lowerbound = round(min_y - delta * 0.1, 2)
    upperbound = round(max_y + delta * 0.1, 2)
    if delta > 5:
        lowerbound = int(lowerbound)
        upperbound = int(upperbound)

    # markers = [
    #     opts.MarkPointItem(
    #         coord=[date, price - (delta * 0.08 if action == 'BUY' else 0)],
    #         value=action,
    #         symbol_size=45 if action == 'BUY' else 60,
    #         symbol="diamond" if action == 'BUY' else "pin",
    #         itemstyle_opts=opts.ItemStyleOpts(color="green" if action == 'BUY' else "red"),
    #     ) for date, price, action in zip(dates, closing_prices, actions) if action in ['BUY', 'SELL']
    # ]
    markers = []
    last_action = None
    for date, price, action in zip(dates, closing_prices, actions):
        if (last_action is None) or (action != last_action):
            point = opts.MarkPointItem(
                coord=[date, price - (delta * 0.08 if action != 'short' else 0)],
                value=action,
                symbol_size=45 if action != 'short' else 60,
                symbol="diamond" if action == 'long' else ("circle" if action == 'close' else "pin"),
                itemstyle_opts=opts.ItemStyleOpts(color="green" if action == 'long' else ("grey" if action == 'close' else "red")),
            )
            markers.append(point)
        last_action = action
        
    if now_date:
        index = dates.index(now_date)
        closing_price_at_now_date = closing_prices[index]
        markers.append(
            opts.MarkPointItem(
                coord=[now_date, closing_price_at_now_date],
                value=now_date,
                symbol_size=120,
                symbol="pin",
                itemstyle_opts=opts.ItemStyleOpts(color="grey"),
            ))

    signal_line = (
        Line()
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(is_show=False),
            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                min_=lowerbound,
                max_=upperbound,
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            legend_opts=opts.LegendOpts(orient="horizontal", pos_top="2%"),
        )
        .add_xaxis(xaxis_data=dates)
        .add_yaxis(
            series_name="Adj Close Prices",
            y_axis=closing_prices,
            symbol="emptyCircle",
            is_symbol_show=True,
            label_opts=opts.LabelOpts(is_show=False),
            markpoint_opts=opts.MarkPointOpts(data=markers)
        )
    )

    return_line = (
        Line()
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(is_show=False),
            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
                axislabel_opts=opts.LabelOpts(formatter="{value}%"),
            ),
            legend_opts=opts.LegendOpts(orient="horizontal", pos_top="55%"),
        )
        .add_xaxis(xaxis_data=dates)
        .add_yaxis(
            series_name="Cumulative Returns",
            y_axis=returns,
            symbol="emptyCircle",
            is_symbol_show=True,
            label_opts=opts.LabelOpts(is_show=False),
        )
    )

    grid_chart = Grid(
        init_opts=opts.InitOpts(
            width="1000px",
            height="800px",
            animation_opts=opts.AnimationOpts(animation=False),
            bg_color="white"
        )
    )
    grid_chart.add(
        signal_line,
        grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height="40%"),
    )
    grid_chart.add(
        return_line,
        grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="60%", height="35%"),
    )

    make_snapshot(driver, grid_chart.render(), output_path)


def main():
    # configs = parse_args()
    # selected_stock = configs.stock
    # selected_algorithm = configs.algo

    selected_algorithm = "dqn"
    tag = 'generator_adv_agent_quantile_nfsp_ma5'

    # stock_list = ["USO", "GLD", "CORN", "SPY", "QQQ", "IWM", "FXE", "FXY", "FXC"]
    stock_list = ["USO"]

    for stock in stock_list:
        for stage in ['train', 'valid', 'test']:
            plot_graph(stock, os.path.join(CURRENT, "workdir/exp/trading"), selected_algorithm, tag, stage)


# cd FinAgentPrivate/downstream_tasks/rl/trading
# python plot.py
if __name__ == '__main__':
    main()
