# Assignment 2 - Zomato Dataset Preprocessing

## Dataset Description
This assignment uses a Zomato food-ordering dataset with `123,657` rows and `12` columns.

The dataset contains restaurant, food item, rating, votes, city, cuisine, and price information. Important columns used in this assignment are:
- `Restaurant Name`
- `Dining Rating`
- `Delivery Rating`
- `Dining Votes`
- `Delivery Votes`
- `Cuisine`
- `Place Name`
- `City`
- `Best Seller`
- `Votes`
- `Prices`

For this assignment, the dataset is processed into a cleaner format with renamed columns such as:
- `rating`
- `votes`
- `cost`

## Files Included
- `zomato.csv` - dataset used in the pipeline
- `preprocessing.py` - data cleaning and preprocessing
- `encoding.py` - all categorical encoding methods
- `scaling.py` - all scaling methods
- `main.py` - runs the full assignment workflow
- `README.md` - assignment explanation

## Steps Performed

### 1. Data Preprocessing
In `preprocessing.py`:
- loaded the dataset
- cleaned column names
- fixed numeric data types
- handled missing values
- removed duplicate rows
- detected outliers using IQR
- treated outliers using capping
- dropped the irrelevant `item_name` column
- created a simple `price_category` column for ordinal encoding

### 2. Missing Value Handling
- Numerical columns were filled with the **median**
- Categorical columns were filled with the **mode**
- If a categorical column had a very high percentage of missing values, it was filled with **Unknown**

### 3. Categorical Encoding
In `encoding.py`, all required encoding techniques were implemented as separate functions:
- **One-Hot Encoding** on `city`
- **Label Encoding** on `best_seller`
- **Ordinal Encoding** on `price_category`
- **Frequency Encoding** on `cuisine` and `place_name`
- **Target Encoding** on `restaurant_name` using `rating`

### 4. Feature Scaling
In `scaling.py`, the following methods were applied on:
- `rating`
- `votes`
- `cost`

Scaling methods used:
- Min-Max Scaling
- Max Absolute Scaling
- Normalization
- Standardization

### 5. Main File
`main.py` performs the full workflow:
- loads the dataset
- runs preprocessing
- applies encoding
- applies scaling
- saves the final processed dataset as `zomato_processed.csv`

## How to Run
```bash
python main.py
```

## Output
After running the project, the final processed file is created:
- `zomato_processed.csv`
- `results_summary.txt`

## Conclusion

### Which missing value method worked best and why?
For this dataset, **median imputation** worked best for numerical columns because rating, votes, and cost values can be affected by outliers. Median is more stable than mean in such cases. For categorical columns, **mode filling** worked well when the missing count was small, while **Unknown** worked better for `best_seller` because that column had a large number of missing values and filling everything with the mode would be misleading.

### Which encoding technique worked best and why?
For this dataset, no single encoding method is best for every column, but **frequency encoding** was the most practical overall for high-cardinality columns like `cuisine` and `place_name` because it kept the dataset simple and avoided creating too many columns. **One-hot encoding** worked well for low-cardinality nominal data like `city`, and **ordinal encoding** was suitable for the ordered `price_category` feature.

### Which scaling method worked best and why?
Among the scaling methods, **standardization** was the most useful overall because it brought `rating`, `votes`, and `cost` onto a comparable scale while preserving their distribution better than normalization. Min-Max scaling also worked, but it is more sensitive to extreme values.

### Observations from outlier handling
The main outliers were present in vote-related columns and cost. Instead of removing those rows, IQR-based capping reduced the effect of extreme values while keeping all records in the dataset. This was a better choice for a food-ordering dataset because very expensive items or highly voted items may still be genuine observations.

### Observation on skewness transformation
Skewness transformation was not applied in this assignment because the task was kept simple and outlier capping was enough for basic preprocessing.
