#!/bin/bash

# Push
# LinePlot
python plot/make_plot.py --csv_path data_plot/push/exp --rename --color --output "lineplot_bsci_push.png" 
# BoxPlot
python plot/make_boxplot.py --csv_path data_plot/push/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers --output "boxplot_bsci_push.png"
python plot/make_boxplot.py --csv_path data_plot/push/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --output "boxplot_push.png"
python plot/make_boxplot.py --csv_path data_plot/push/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers -y eval/success_rate --output "boxplot_succesrate_bsci_push.png"
python plot/make_boxplot.py --csv_path data_plot/push/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 -y eval/success_rate --output "boxplot_succesrate_push.png"

# Basketball
# LinePlot
python plot/make_plot.py --csv_path data_plot/basketball/exp --rename --color --output "lineplot_bsci_basketball.png" 
# BoxPlot
python plot/make_boxplot.py --csv_path data_plot/basketball/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers --output "boxplot_bsci_basketball.png"
python plot/make_boxplot.py --csv_path data_plot/basketball/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --output "boxplot_basketball.png"
python plot/make_boxplot.py --csv_path data_plot/basketball/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers -y eval/success_rate --output "boxplot_succesrate_bsci_basketball.png"
python plot/make_boxplot.py --csv_path data_plot/basketball/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 -y eval/success_rate --output "boxplot_succesrate_basketball.png"

# bin-picking
# LinePlot
python plot/make_plot.py --csv_path data_plot/bin-picking/exp --rename --color --output "lineplot_bsci_bin-picking.png" 
# BoxPlot
python plot/make_boxplot.py --csv_path data_plot/bin-picking/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers --output "boxplot_bsci_bin-picking.png"
python plot/make_boxplot.py --csv_path data_plot/bin-picking/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --output "boxplot_bin-picking.png"
python plot/make_boxplot.py --csv_path data_plot/bin-picking/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers -y eval/success_rate --output "boxplot_succesrate_bsci_bin-picking.png"
python plot/make_boxplot.py --csv_path data_plot/bin-picking/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 -y eval/success_rate --output "boxplot_succesrate_bin-picking.png"

# button-press
# LinePlot
python plot/make_plot.py --csv_path data_plot/button-press/exp --rename --color --output "lineplot_bsci_button-press.png" 
# BoxPlot
python plot/make_boxplot.py --csv_path data_plot/button-press/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers --output "boxplot_bsci_button-press.png"
python plot/make_boxplot.py --csv_path data_plot/button-press/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --output "boxplot_button-press.png"
python plot/make_boxplot.py --csv_path data_plot/button-press/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers -y eval/success_rate --output "boxplot_succesrate_bsci_button-press.png"
python plot/make_boxplot.py --csv_path data_plot/button-press/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 -y eval/success_rate --output "boxplot_succesrate_button-press.png"

# shelf-place
# LinePlot
python plot/make_plot.py --csv_path data_plot/shelf-place/exp --rename --color --output "lineplot_bsci_shelf-place" 
# BoxPlot
python plot/make_boxplot.py --csv_path data_plot/shelf-place/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers --output "boxplot_bsci_shelf-place"
python plot/make_boxplot.py --csv_path data_plot/shelf-place/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --output "boxplot_shelf-place"
python plot/make_boxplot.py --csv_path data_plot/shelf-place/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers -y eval/success_rate --output "boxplot_succesrate_bsci_shelf-place"
python plot/make_boxplot.py --csv_path data_plot/shelf-place/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 -y eval/success_rate --output "boxplot_succesrate_shelf-place"

# All
# LinePlot
python plot/make_plot.py --csv_path data_plot/all/exp --rename --color --output "lineplot_bsci_all.png" 
# BoxPlot
python plot/make_boxplot.py --csv_path data_plot/all/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers --output "boxplot_bsci_all.png"
python plot/make_boxplot.py --csv_path data_plot/all/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --output "boxplot_all.png"
python plot/make_boxplot.py --csv_path data_plot/all/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 --baseline_whiskers -y eval/success_rate --output "boxplot_succesrate_bsci_all.png"
python plot/make_boxplot.py --csv_path data_plot/all/exp --rename --color --baseline --custom_order --box_type std_err --whiskers_percentile 90 --fixed_x 100000 --y_min 0 -y eval/success_rate --output "boxplot_succesrate_all.png"
