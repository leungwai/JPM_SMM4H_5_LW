# 🐤 JPM_SMM4H_5_LW

**Task 5** for the **7th Social Mining Media for Health (SMM4H)** competition hosted by **International Conference on Computational Logistics (COLING) 2022**

_by Adrian Garcia Hernandez and Leung Wai Liu_

This repo is training, ensembling and analysis code for the BERT Model used for Task 5 of the SMM4H competition that I competed in during my internship at the AI Research team in J.P. Morgan Chase in the Summer of 2022. 

_\#NLP \#BERT \#ML \#Python_

**See Also:** [Subtask 1a](https://github.com/leungwai/JPM_SMM4H_1a_LW) | [Subtask 1b](https://github.com/leungwai/JPM_SMM4H_1b_LW) | [Subtask 1c](https://github.com/leungwai/JPM_SMM4H_1c_LW) | [Subtask 2a](https://github.com/leungwai/JPM_SMM4H_2a_LW) | [Subtask 2b](https://github.com/leungwai/JPM_SMM4H_2b_LW) 

## Premise
The need to use Natural Language Processing \(NLP\) on social media posts is increasingly important as its userbase grows to guage public perception on issues, such as sentiments during the COVID-19 pandemic. 

## Task Description
Task 5 is a three-way classification problem where a Spanish-language tweet has to be identified as one of three possible classess: Self Reports, Non-Personal Reports, and Literature News Mentions. 

## Methodology
The datasets were trained on variants top of the BERT language model \(Devlin et al., 2019\): cased and uncased BERT<sub>BASE</sub>-multilingual, Spanish-BERT(Cañete et al., 2020), and XLM-RoBERTa (Conneau et al., 2019).

The model ensembling methodology is adapted from \(Jayanthi and Gupta, 2021\) method of model ensembling. Various methods of ensembling were experimented, including majority-vote, weighted and unweighted. Ultimately, a majority vote ensemble of RoBERTa<sub>LARGE</sub> models were used. 

## Results 
**Performance Metric for Task 5**
| Task | F1-Score |
| ---: | :---: |
| Task 5 | 0.85 | 

> Tied **2nd** of 7 submissions

## Special Thanks
- **Akshat Gupta**, for being a great project manager and guiding us through NLP from start to finish
- **Saheed Obitayo**, for being a great manager
- The organizers for the 7th SMM4H competition and 2022 COLING conference
