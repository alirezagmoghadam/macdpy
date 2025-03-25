## Code for MACD Enhancement through Optimized Weighting and Stability Analysis


Table of contents
=================

<!--ts-->
  * [Background](#Background)
  * [Overview](#Overview)
  * [Environment setup](#Environment-setup)
  * [Dataset](#Dataset)
  * [Run](#Run)

<!--te-->

Background
============
To make profits, capital market participants need to schedule correctly to exit and enter positions at perfect times, where finding the optimal time can be daunting. One of the technical indicators for different financial markets is moving average convergence divergence (MACD), which traders use to measure the performance and efficiency of the market and increase profit. In this paper, we introduce two new speeded and correntropy-based versions of MACD, which propose new mechanisms to boost the performance and stability of MACD and boost profit. Additionally, we proved the stability of the proposed methods for better performance evaluation. To avoid concerns over bias, we report the results over validation periods and deploy the systems on 1-day, 1-hour, and 5-minute time formats of financial securities. Our results show that using technical analysis strategies based on the original MACD indicator results in a low success rate of around 30% for individually tested stocks in the Dow group because of late and false signals associated with the indicator and in all benchmarks, the proposed methods perform better than MACD, and in some periods, the system outperforms the holding of the same stocks.

![](Figure1_MACD_Diagram.png)
![](Figure3_SMACD_Diagram.png)
![](Figure6_CMACD_Diagram.png)


Environment setup
==================

    git clone https://github.com/alirezagmoghadam/macdpy.git
    cd macdpy
    conda env create -f environment.yml
    conda activate macdpy

Dataset
-------------

The dataset was acquired from [Polygon.io](https://polygon.io/). Contact the author for both the data and the results.

Run
-----------------

To run the code, run the experiment_main.py file.
