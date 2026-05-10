import pandas as pd #import pandas library
import numpy as np #import numpy library
import matplotlib.pyplot as plt #import matplotlib library
import seaborn as sns #import seaborn library
from scipy import stats #import scipy library

# Load the dataset
df=pd.read_csv('Global_Music_Streaming_Listener_Preferences.csv')
print(df) 

# Get shape (rows, columns)
print("Shape of dataset:", df.shape)

# Dataset Summary - Check Data Types
print(df.info()) 

# Stastical Summary - Gives Stastical Parameters
print(df.describe()) 

# Detect Missing Value - Identify Missing Value
print(df.isnull()) 

# Identify Outliers Using IQR Method

# Calculate the first quartile (Q1) → 25% of the data lies below this value
Q1 = df['Age'].quantile(0.25)

# Calculate the third quartile (Q3) → 75% of the data lies below this value
Q3 = df['Age'].quantile(0.75)

# Calculate Interquartile Range (IQR) → difference between Q3 and Q1
IQR = Q3 - Q1

# Calculate lower limit → values below this are considered outliers
lower_limit = Q1 - 1.5 * IQR

# Calculate upper limit → values above this are considered outliers
upper_limit = Q3 + 1.5 * IQR

# Filter dataset to find outliers
# Condition:
# Age < lower_limit OR Age > upper_limit
outliers = df[(df['Age'] < lower_limit) | (df['Age'] > upper_limit)]

# Display all outlier rows
print(outliers)

#Remove Duplicates
print("Before removing duplicates:", df.shape)

df = df.drop_duplicates(subset=[
    "Age",
    "Country",
    "Top Genre",
    "Most Played Artist",
    "Subscription Type",
    "Listening Time (Morning/Afternoon/Night)"
])

print("After removing duplicates:", df.shape)

#---------------------------------------------------------------------------
#                          NUMERICAL OPERATIONS
#---------------------------------------------------------------------------

#...................BASIC NUMERICAL OPERATIONS...........
# addition
# Create a new column 'Total Activity'
# It adds Minutes Streamed and Number of Songs Liked
df['Total Activity'] = df['Minutes Streamed Per Day'] + df['Number of Songs Liked']

# Display selected columns to see result
print(df[['Minutes Streamed Per Day', 'Number of Songs Liked', 'Total Activity']].head())

# subtraction
# Create a new column showing difference between streaming and liking
df['Activity Difference'] = df['Minutes Streamed Per Day'] - df['Number of Songs Liked']

# Print first few values of the new column
print(df[['Activity Difference']].head())

# multiplication
# Multiply number of songs liked with repeat song rate percentage
# This shows how actively a user repeats songs they like
df['Interaction Score'] = df['Number of Songs Liked'] * df['Repeat Song Rate (%)']

# Display relevant columns to understand the result
print(df[['Number of Songs Liked', 'Repeat Song Rate (%)', 'Interaction Score']].head())

# division
# Divide number of songs liked by minutes streamed
# This shows how frequently users like songs
df['Like Ratio'] = df['Number of Songs Liked'] / df['Minutes Streamed Per Day']

# Print the ratio values
print(df[['Like Ratio']].head())

# modulus
# Find remainder when minutes streamed is divided by songs liked
df['Remainder'] = df['Minutes Streamed Per Day'] % df['Number of Songs Liked']

# Display the remainder values
print(df[['Remainder']].head())

# power
# Square the streaming minutes (raise to power 2)
df['Squared Streaming'] = df['Minutes Streamed Per Day'] ** 2

# Show original and squared values
print(df[['Minutes Streamed Per Day', 'Squared Streaming']].head())

#...................BASIC STATISTICAL OPERATIONS.........
# mean
# Calculate average minutes streamed per day
mean_value = df['Minutes Streamed Per Day'].mean()

# Print the result
print("Average Minutes Streamed:", mean_value)

# median
# Find the middle value of streaming minutes
median_value = df['Minutes Streamed Per Day'].median()

# Print result
print("Median Minutes Streamed:", median_value)

# mode
# Find most common genre
mode_value = df['Top Genre'].mode()

# Print result
print("Most Common Genre:", mode_value)

# standard deviation
# Calculate standard deviation of streaming minutes
std_value = df['Minutes Streamed Per Day'].std()

# Print result
print("Standard Deviation:", std_value)

# max-min age
# Find minimum age
min_age = df['Age'].min()

# Find maximum age
max_age = df['Age'].max()

# Print results
print("Minimum Age:", min_age)
print("Maximum Age:", max_age)

# count
# Count total non-null values in 'Country' column
count_value = df['Country'].count()

# Print result
print("Total Country Entries:", count_value)

# correlation
# Find relationship between numeric columns
correlation = df.corr(numeric_only=True)

# Print correlation matrix
print(correlation)

#...................DATA MANIPULATION OPERATIONS.........

# Filter users who have Premium subscription
premium_users = df[df['Subscription Type'] == 'Premium']

# Display result
print(premium_users.head())

# Filter users who are under 30 AND stream more than 300 minutes
filtered_data = df[(df['Age'] < 30) & (df['Minutes Streamed Per Day'] > 300)]

# Show result
print(filtered_data.head())

# Sort dataset by streaming minutes in descending order
sorted_data = df.sort_values(by='Minutes Streamed Per Day', ascending=False)

# Display top rows
print(sorted_data.head())

# Group data by Country and calculate average streaming minutes
group_data = df.groupby('Country')['Minutes Streamed Per Day'].mean()

# Print grouped result
print(group_data)


#---------------------------------------------------------------------------
#                          DATA DISTRIBUTION
#---------------------------------------------------------------------------

#....................AGE DISTRIBUTION (Histrogram plot)....................

# Convert 'Age' column into NumPy array (for numerical operations)
age = np.array(df['Age'])

# Plot histogram of age
# bins=10 → divide data into 10 groups
# density=True → show probability density instead of count
# alpha=0.6 → transparency of bars
plt.hist(age, bins=10, density=True, alpha=0.6)

# Fit a normal distribution (Gaussian curve) to the data
# mu = mean, std = standard deviation
mu, std = stats.norm.fit(age)

# Get current x-axis limits of the plot
xmin, xmax = plt.xlim()

# Create evenly spaced values between xmin and xmax
x = np.linspace(xmin, xmax, 100)

# Calculate probability density function for normal distribution
p = stats.norm.pdf(x, mu, std)

# Plot the normal distribution curve
plt.plot(x, p, linewidth=2)

# Add title and labels
plt.title("Age Distribution of Listeners")
plt.xlabel("Age")
plt.ylabel("Density")

# Display the plot
plt.show()

#......................GENRE DISTRIBUTION (Bar plot)...................

# Extract 'Top Genre' column
genres = df['Top Genre']

# Get unique genres and their counts
unique_genres, counts = np.unique(genres, return_counts=True)

# Plot bar chart
plt.bar(unique_genres, counts)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add labels and title
plt.title("Genre Distribution")
plt.xlabel("Top Genre")
plt.ylabel("Number of Listeners")

# Show plot
plt.show()

#.....................SUBSCRIPTION DISTRIBUTION (Pie Chart)........................

# Extract subscription column
subscription = df['Subscription Type']

# Get unique values and counts
unique_sub, counts = np.unique(subscription, return_counts=True)

# Plot pie chart
plt.pie(counts, labels=unique_sub, autopct='%1.1f%%')

# Add title
plt.title("Subscription Type Distribution")

# Show plot
plt.show()

#.....................MINUTES STREAMED PER DAY DISTRIBUTION (Histogram plot).............

# Convert to array
minutes = np.array(df['Minutes Streamed Per Day'])

# Plot histogram
plt.hist(minutes, bins=10, density=True, alpha=0.6)

# Fit normal distribution
mu, std = stats.norm.fit(minutes)

# Generate x values
x = np.linspace(min(minutes), max(minutes), 100)

# Plot curve
plt.plot(x, stats.norm.pdf(x, mu, std), linewidth=2)

# Labels
plt.title("Distribution of Minutes Streamed Per Day")
plt.xlabel("Minutes")
plt.ylabel("Density")

# Show
plt.show()

#.....................NUMBER OF SONGS LIKED (Histogram plot).............................

# Convert to array
likes = np.array(df['Number of Songs Liked'])

# Plot histogram
plt.hist(likes, bins=10, density=True, alpha=0.6)

# Fit normal distribution
mu, std = stats.norm.fit(likes)

# Generate x values
x = np.linspace(min(likes), max(likes), 100)

# Plot curve
plt.plot(x, stats.norm.pdf(x, mu, std), linewidth=2)

# Labels
plt.title("Distribution of Number of Songs Liked")
plt.xlabel("Number of Songs Liked")
plt.ylabel("Density")

# Show
plt.show()

#...................REPEAT SONGS RATE DISTRIBUTION (Histogram plot)......................

# Convert to array
repeat_rate = np.array(df['Repeat Song Rate (%)'])

# Plot histogram
plt.hist(repeat_rate, bins=10, density=True, alpha=0.6)

# Fit normal distribution
mu, std = stats.norm.fit(repeat_rate)

# Generate x values
x = np.linspace(min(repeat_rate), max(repeat_rate), 100)

# Plot curve
plt.plot(x, stats.norm.pdf(x, mu, std), linewidth=2)

# Labels
plt.title("Distribution of Repeat Song Rate")
plt.xlabel("Repeat Rate")
plt.ylabel("Density")

# Show
plt.show()

#...................Genre vs Listening Time (Heatmap plot)......................
pivot = df.pivot_table(
    index="Top Genre",
    columns="Listening Time (Morning/Afternoon/Night)",
    values="Minutes Streamed Per Day",
    aggfunc="mean"
)

sns.heatmap(pivot, annot=True, cmap="YlGnBu")
plt.title("Genre vs Listening Time (Avg Minutes)")
plt.show()

#...................Streaming Behavior by Subscription Type (Violin Plot)......................
plt.figure(figsize=(6,4))

sns.violinplot(data=df, x="Subscription Type", y="Minutes Streamed Per Day", palette="Set2")

plt.title("Distribution of Streaming Time by Subscription Type")
plt.xlabel("Subscription Type")
plt.ylabel("Minutes Streamed Per Day")

plt.show()

#...................Streaming Time Density (KDE Plot)......................
plt.figure(figsize=(6,4))

sns.kdeplot(data=df, x="Minutes Streamed Per Day", hue="Subscription Type", fill=True)

plt.title("Density Distribution of Streaming Time by Subscription Type")
plt.xlabel("Minutes Streamed Per Day")
plt.ylabel("Density")

plt.show() 

#..................Country-wise User Distribution (CountPlot)..............
plt.figure(figsize=(6,5))

order = df['Country'].value_counts().index

sns.countplot(data=df, y="Country", order=order)

plt.title("Number of Users by Country")
plt.xlabel("Count")
plt.ylabel("Country")

plt.show()

#..................Engagement Score Analysis (Box Plot)....................
plt.figure(figsize=(6,4))

sns.boxplot(data=df, x="Subscription Type", y="Interaction Score")

plt.title("User Engagement Score by Subscription Type")
plt.xlabel("Subscription Type")
plt.ylabel("Interaction Score")

plt.show()


#---------------------------------------------------------------------------
#                          DATA NORMALIZATION
#---------------------------------------------------------------------------

#.............. Min-Max Scaling - Normalize Minutes Streamed (0 to 1 scale) ...................

df['Minutes_Normalized'] = (
    (df['Minutes Streamed Per Day'] - df['Minutes Streamed Per Day'].min()) /
    (df['Minutes Streamed Per Day'].max() - df['Minutes Streamed Per Day'].min())
)

print(df[['Minutes Streamed Per Day', 'Minutes_Normalized']].head())

#---------------------------------------------------------------------------
#                          DATA STANDARDIZATION
#---------------------------------------------------------------------------

#.............. Z-score Scaling - Standardize Age .........................
df['Age_Standardized'] = (
    (df['Age'] - df['Age'].mean()) / df['Age'].std()
)

print(df[['Age', 'Age_Standardized']].head())

#---------------------------------------------------------------------------
#                          DATA ENCODING
#---------------------------------------------------------------------------

#................ Label Encoding - Convert Subscription Type into numbers....................
df['Subscription_Encoded'] = df['Subscription Type'].map({
    'Free': 0,
    'Premium': 1
})

print(df[['Subscription Type', 'Subscription_Encoded']].head())

#---------------------------------------------------------------------------
#                          DATA VISUALIZATION
#---------------------------------------------------------------------------
import matplotlib.pyplot as plt

listening_count = df['Listening Time (Morning/Afternoon/Night)'].value_counts()

plt.bar(listening_count.index, listening_count.values)
plt.title("Number of Listeners by Time of Day")
plt.xlabel("Listening Time")
plt.ylabel("Number of Listeners")
plt.show()