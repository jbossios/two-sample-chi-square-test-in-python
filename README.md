# Example of a two-sample chi-square test in Python

## Dependencies

```
numpy
pandas
scipy
```

## Introduction

Let's imagine we have a hotel booking website and wish to study if a given change in our website can boost our conversion rates (at the final stage of the booking process). We decide then to make an A/B test to help us determine if we want to release such a change. For this example, let's set the significance level to 0.05 (alpha) and the statistical power (1-beta) to 0.8 (the statistical power will be used to define the minimum sample size).

In this example, our null hypothesis states that there is no significant difference between the conversion rates with or without such a change in the website.

In ```example.py```, you will find the implementation of a two-sample chi-square test, including the generation of fake data as well as the determination of the minimum sample size.

*If you like this course, please consider giving me a star!*

## How to run full example?

Run the following:

```
python example.py
```

## Extra

If you wish to learn how to perform a data analysis in Python, check out my free course [here](https://github.com/jbossios/python-tutorial).
