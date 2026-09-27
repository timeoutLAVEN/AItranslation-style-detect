# AItranslation-style-detect
This project mainly aims to verify that the RS-method has the ability to detect text style across machine translation models in four languages—Russian, Chinese, English, and Amharic—and that it is robust in the sense that it does not transfer with compression algorithms, compression sizes, or compression content.

test19onefile is the MATLAB running code. The files below whose suffixes are marked with the translation model are the raw translation materials. The translation materials come from the online platform https://sider.ai/zh-CN/translator/text-translator. The running results are shown in test6mateerials.xlsx, in which the stability of CramerV is demonstrated (repeated experiments will not change the results).

I would like to express my sincere gratitude to my colleague, Lulu Yeshewas, for his invaluable contributions to the Amharic language. His link is provided below for your reference:https://github.com/yeshewas/Assessing-the-quality-of-some-machine-translation-systems-using-an-information-theoretic-RS-method

In order to check whether the table data is incorrect (related to the calculation of Cramer's V coefficient), I wrote a MATLAB script called checkCramer.m, which takes a contingency table and the original Cramer's V value as input. We will calculate the Cramer's V coefficient and verify whether it is consistent with the original table result.
