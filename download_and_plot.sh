#!/bin/bash
sh data_plot/exp_multiheads_15/data_download.sh
sh data_plot/exp_multiheads_30/data_download.sh
sh data_plot/exp_multiheads_35/data_download.sh

sh data_plot/exp_multiheads_15/barplot.sh
sh data_plot/exp_multiheads_30/barplot.sh
sh data_plot/exp_multiheads_35/barplot.sh


sh data_plot/exp_multiheads_20/data_download.sh
sh data_plot/exp_multiheads_10/data_download.sh
sh data_plot/exp_multiheads_20/barplot.sh
sh data_plot/exp_multiheads_10/barplot.sh

# cp data_plot/exp_big/all/none* data_plot/exp_multiheads_35/all/
# cp data_plot/exp_big/basketball/none* data_plot/exp_multiheads_35/basketball/
# cp data_plot/exp_big/bin-picking/none* data_plot/exp_multiheads_35/bin-picking/
# cp data_plot/exp_big/button-press/none* data_plot/exp_multiheads_35/button-press/
# cp data_plot/exp_big/push/none* data_plot/exp_multiheads_35/push/
# cp data_plot/exp_big/shelf-place/none* data_plot/exp_multiheads_35/shelf-place/

# cp data_plot/exp_multiheads_15/basketball/none*   data_plot/exp_multiheads_15/all/
# cp data_plot/exp_multiheads_15/bin-picking/none*  data_plot/exp_multiheads_15/all/
# cp data_plot/exp_multiheads_15/button-press/none* data_plot/exp_multiheads_15/all/
# cp data_plot/exp_multiheads_15/push/none*         data_plot/exp_multiheads_15/all/
# cp data_plot/exp_multiheads_15/shelf-place/none*  data_plot/exp_multiheads_15/all/