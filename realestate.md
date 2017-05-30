## Real Estate Price Index

Three methods:

1. median method
2. hedonic method
3. repeat-sales method

### Case-Shiller Index

References:

1. https://en.wikipedia.org/wiki/Case%E2%80%93Shiller_index
2. http://www.nber.org/papers/w2393

The index will be published with a 2-month lag on the last Tuesday of every month.

Shiller believes that there's no continuous uptrend in home prices in the US and the
home prices show a strong tendency to return to its 1890 level in teal terms.

The model is an improvement to Bailey(1963)'s repeat sales model.

### Zillow's approach

get the price estimate for each property to form the house-level index, and then aggregate based on
median price to form the index for a given region

https://www.zillow.com/research/zhvi-methodology-6032/

To handle the sparsity of the transactions for one house, Zillow valuates the price index for each property
every t time, which then be used in Zillow Home Value Index model.

There're some good discussions available in kaggle's zillow competition.

https://www.kaggle.com/c/zillow-prize-1


### Problems for the current models/methods

1. Case-Shiller index or other repeat sales model is good when fitting to a coarse degree, i.e something like
metropolitan areas or cities, while when it comes for fine areas, the data will become very sparse, and hard to
make it very precise
