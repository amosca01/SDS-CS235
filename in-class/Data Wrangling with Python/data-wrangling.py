
# coding: utf-8

# # Introduction to Pandas: the Python Data Analysis library
# This is a short introduction to pandas, geared mainly for new users and adapted heavily from the "10 Minutes to Pandas" tutorial from http://pandas.pydata.org. You can see more complex recipes in the Pandas Cookbook: http://pandas.pydata.org/pandas-docs/stable/cookbook.html
# ## Initial setup
# Let's start by importing 2 useful libraries: `pandas` and `numpy`

# In[2]:

import pandas, numpy


# You can create a `Series` by passing a list of values. By default, `pandas` will create an integer index.

# In[3]:

s = pandas.Series([1,3,5,"CSC is awesome",8])
s


# You can also use `pandas` to create an series of `datetime` objects. Let's make one for the week beginning March 1st, 2021:

# In[4]:

dates = pandas.date_range('20210301',
                          periods = 7)
dates


# Now we'll create a `DataFrame` using the `dates` array as our index, fill it with some random values using `numpy`, and give the columns some labels.

# In[5]:

df = pandas.DataFrame(numpy.random.randn(7,4), 
                      index = dates, 
                      columns = {'dog','cat','mouse','duck'})
df


# It can also be useful to know how to create a `DataFrame` from a `dict` of objects. This comes in particularly handy when working with JSON-like structures.

# In[6]:

df2 = pandas.DataFrame({ 'A' : 1.,
                     'B' : pandas.Timestamp('20130102'),
                     'C' : pandas.Series(1,index=list(range(5)),dtype='float32'),
                     'D' : numpy.array([3] * 5
                                    ,dtype='int32'),
                     'E' : pandas.Categorical(["test","train","blah","train","blah"]),
                     'F' : 'foo' })
df2


# ## Exploring the data in a DataFrame
# We can access the data types of each column in a `DataFrame` as follows:

# In[7]:

df2.dtypes


# We can display the `index`, `columns`, and the underlying `numpy values` separately:

# In[8]:

df2.index


# In[9]:

df2.columns


# In[10]:

df2.values


# To get a quick statistical summary of your data, use the `.describe()` method:

# In[11]:

df2.describe()


# ## Some basic data transformations
# `DataFrames` have a built-in transpose:

# In[12]:

df.T


# We can also sort a `DataFrame` along a given data dimension. For example, we might want to `sort` by the values in the `duck` column:

# In[13]:

df.sort_values(by = "cat")


# We can also sort the rows `(axis=0)` and columns `(axis=1)` by their index/header values:

# In[14]:

df.sort_index(axis = 0, 
              ascending = False)


# In[15]:

df.sort_index(axis = 1)


# ## Selection
# To see only only the first few rows of a `DataFrame`, use the `.head()` function:

# In[16]:

df.head()


# To view only the last few rows, use the `.tail()` function. Note that by default, both `.head()` and `.tail()`$ return 5 rows. You can also specify the number you want by passing in an integer.

# In[17]:

df.tail(3)


# Selecting a single column via indexing yields a `Series`:

# In[18]:

df2['A'] # a lot like df$A in R


# We can also select a subset of the rows using slicing. You can select either by integer indexing:

# In[19]:

df[2:5]


# To select more than one column at a time, try `.loc[]`:

# In[20]:

df.loc[:, ['cat', 'dog']]


# And of course, you might want to do both at the same time:

# In[21]:

df.loc['20210302':'20210305', ['cat', 'duck']]


# ## Boolean Indexing
# Sometimes it's useful to be able to select all rows that meet some criteria. For example, we might want all rows where the value in the `cat` column is greater than 0:

# In[22]:

df[df['cat'] > 0]


# Or perhaps we'd like to eliminate all negative values:

# In[23]:

nonneg = df[df > 0]
nonneg


# And then maybe we'd like to drop all the rows with missing values:

# In[24]:

nonneg.dropna()


# Oops... maybe not. How about we set them to 0 instead?

# In[25]:

nonneg.fillna(value = 0)


# But what if your values aren't numeric? No problem, we can also do filtering. First, let's copy the `DataFrame` and add a new column of nominal values:

# In[26]:

df3 = df.copy()
df3['color'] = ['blue', 'green','red','blue','green','red','blue']
df3


# Now we can use the `.isin()` function to select only the rows with `green` or `blue` in the `color` column:

# In[27]:

df3[df3['color'] == 'green']


# In[28]:

df3 # Note that none of the filtering operations change the original DataFrame


# ## Prefer something more like `dplyr`?
# The `dfply` module provided dplyr-style piping operations for `pandas DataFrames`, along with many familiar verbs. Note that `>>` is the `dfply` equivalent of `%>%`, and `X` represents the piped `DataFrame`.
# 
# ### `select()`

# In[29]:

from dfply import *
df2 >> select("A")


# Just be careful about python and spacing:

# In[30]:

# Broken
# df2 >> 
#    select(X.A)
    
# If you want to indent, you need parens
(
df2 >> 
   select(X.A)
)


# ### Dropping columns with `drop()` or `select()`
# 
# `drop()` is a helper function that does the opposite of `select()`:

# In[31]:

(df2 >> 
   drop(X.E, X.F))


# You can also use `select()` along with the `~` (which means "not") to drop unwanted columns:

# In[32]:

(df2 >> 
   select(~X.E, ~X.F))


# ### Filtering with `mask()`
# `mask()` keeps all the rows where the criteria is/are true:

# In[33]:

# this is just like filter() in R
(df3 >>
   mask(X.color.isin(['green','blue']), X.cat > 0))


# ### Add new columns with `mutate()`

# In[34]:

(df3 >>
   mutate(platypus = (X.cat + X.duck)/2))


# ## Basic Math
# Helper methods on the `DataFrame` object make it straightforward to calculate things like the `.mean()` across all numeric columns:

# In[35]:

df.mean(axis = 0)


# We can also perform the same operation on individual rows:

# In[36]:

df.mean(axis = 1)


# `.median()` also behaves as expected:

# In[37]:

df.median(axis = 0)


# No helper function for what you need? No problem. You can also use the `.apply()` function to evaluate functions to the data. For example, we might want to perform a cumulative summation (thanks, `numpy`!):

# In[38]:

df.apply(numpy.cumsum)


# Or apply your own function, such as finding the spread (`max` value - `min` value):

# In[39]:

df.apply(lambda x: x.max() - x.min(), axis = 0)


# ## Combining DataFrames
# Combining `DataFrame` objects can be done using simple concatenation (provided they have the same columns):

# In[40]:

frame_one = pandas.DataFrame(numpy.random.randn(5, 4))
frame_one


# In[41]:

frame_two = pandas.DataFrame(numpy.random.randn(5, 4))
frame_two


# In[42]:

pandas.concat([frame_one, frame_two])


# If your `DataFrames` do not have an identical structure, but do share a common key, you can also perform a SQL-style join using the `.merge()` function:

# In[43]:

left = pandas.DataFrame({'blah': ['foo', 'bar'], 
                         'lval': [1, 2]})
left


# In[44]:

right = pandas.DataFrame({'blah': ['foo', 'foo', 'bar'], 
                          'rval': [3, 4, 5]})
right


# In[45]:

joined_data = pandas.merge(left, right, on = "blah")
joined_data


# Don't worry, `dfply` lets us do this using `join`s too:

# In[46]:

left >> inner_join(right, by = "blah")


# ## Grouping
# Sometimes when working with multivariate data, it's helpful to be able to condense the data along a certain dimension in order to perform a calculation for efficiently. Let's start by creating a somewhat messy `DataFrame`:

# In[47]:

foo_bar = pandas.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
                            'B' : ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                            'C' : numpy.random.randn(8),
                            'D' : numpy.random.randn(8)})
foo_bar


# Now let's group by column `A`, and `sum()` along the other columns:

# In[48]:

foo_bar.groupby('A').sum()


# Note that column `B` was dropped, because the summation operator doesn't make sense on strings. However, if we wanted to retain that information, we could perform the same operation using a hierarchical index:

# In[49]:

grouped = foo_bar.groupby(['A','B']).sum()
grouped


# The `stack()` function can be used to "compress” a level in the `DataFrame`’s columns:

# In[50]:

stacked = grouped.stack()
stacked


# To uncompress the last column of a stacked `DataFrame`, you can call `.unstack()`:

# In[51]:

stacked.unstack()


# ## Time Series
# `pandas` has simple, powerful, and efficient functionality for performing resampling operations during frequency conversion (for example, converting secondly data into minutely data). Firse, let's create an array of 150 `dateTime` objects at a frequency of 1 second:

# In[52]:

rng = pandas.date_range('1/1/2021', 
                        periods = 350, 
                        freq = 'S')
rng


# Now we'll use that to greate a time series, assigning a random integer to each element of the range:

# In[53]:

time_series = pandas.Series(numpy.random.randint(0, 500, len(rng)), 
                            index = rng)
time_series.head()


# Next, we'll resample the data by binning the one-second raw values into minutes (and summing the associated values):

# In[54]:

time_series.resample('1Min').sum()


# We also have support for time zone conversion. For example, if we assume the original `time_series` was in UTC:

# In[55]:

ts_utc = time_series.tz_localize('UTC')
ts_utc.head()


# We can easily convert it to Eastern time:

# In[56]:

ts_utc.tz_convert('US/Eastern').head()


# ## Reading/Writing to files
# Writing to a file works just like we'd expect:

# In[57]:

ts_utc.to_csv("foo.csv")


# As does reading:

# In[58]:

new_frame = pandas.read_csv("foo.csv")
new_frame.head()


# We can also read/write to `.xlsx` format, if for some reason we want to work in Excel:

# In[59]:

ts_utc.to_excel("foo.xlsx", sheet_name = "Sample 1")


# But what if the data is a little... messy? Something like this:

# In[60]:

broken_df = pandas.read_csv('bikes.csv')


# No problem! The `read_csv()` function has lots of tools to help wrangle this mess. Here we'll
# 
#     - change the column separator to a ;
#     - Set the encoding to `latin1` (the default is `utf8`)
#     - Parse the dates in the `Date` column
#     - Tell it that our dates have the day first instead of the month first

# In[61]:

fixed_df = pandas.read_csv('bikes.csv', 
                           sep = ';', 
                           encoding = 'latin1', 
                           parse_dates = ['Date'], 
                           dayfirst = True)
fixed_df.head()


# ## Scraping Data
# Many of you will probably be interested in scraping data from the web for your projects. For example, what if we were interested in working with some data on municipalities in the state of Massachusetts? Well, we can get that from https://en.wikipedia.org/wiki/List_of_municipalities_in_Massachusetts
# 
# First, we'll import two useful libraries: `requests` (to query a URL) and `BeautifulSoup` (to parse the web page):

# In[62]:

import requests
from bs4 import BeautifulSoup


# Now let's request the page:

# In[63]:

result = requests.get("https://en.wikipedia.org/wiki/List_of_municipalities_in_Massachusetts")
result


# The response code `[200]` indicates that the request was successful. Now let's use `BeautifulSoup` to parse the web page into something semi-readable:

# In[64]:

page = result.content
soup = BeautifulSoup(page)
print(soup.prettify())


# Notice that there's a lot of code in there that isn't related to the data we're trying to get. If we inspect the page source of the table we're interested in, we see that it's a `<table>` elemnt with `class = wikitable sortable`. Let's ask `BeautifulSoup` to give us just that piece of the page:

# In[65]:

table = soup.find("table", { "class" : "wikitable sortable" })
print(table.prettify())


# Excellent! Now we just need to pull out the data from the rows and columns of that table. Notice that there are 6 columns, each `row` is contained in a `<tr>` element (which stands for `table row`), and inside each row there is a `<td>` element for each entry. That's what we want, so let's loop through them:

# In[66]:

import re # Might need to do some pattern matching with regular expressions

# Open a file to store the data
f = open('output.csv', 'w')
 
# First grab the headers
t_headers = []
for th in table.find_all("th"):
        # remove any references in square brackets and extra spaces from left and right
        t_headers.append(re.sub('\[.*?\]', '', th.text).strip())

f.write(",".join(t_headers) + "\n")
    
# Loop over each row in the table
for row in table.findAll("tr"):
    
    # Find all the cells in the row
    cells = row.findAll("td")
    
    # If the row is complete (i.e. there are 7 cells)
    #    assign the value in each cell to the appropriate variable.
    if len(cells) == 7:
        
        Municipality = cells[0].find(text = True)
        
        Type = cells[1].find(text = True)
        
        County = cells[2].find(text = True)
        
        Form_of_government = cells[3].find(text = True)
        
        # This cell sometimes has a comma in it, let's get rid opf that while we're here
        Population = cells[4].find(text = True).replace(',', '')
        
        Area = cells[5].find(text = True)
        
        Year_est = cells[6].find(text = True)
 
        # Concatenate all the cells together, separated by commas
        line = Municipality  + "," + Type + "," + County + "," + Form_of_government + "," + Population + "," + Area + "," + Year_est + "\n"
        
        # Append the line to the file
        f.write(line)
        
# Clean up when we're done
f.close()


# ## And there you have it! You're ready to wrangle some data of your own.
