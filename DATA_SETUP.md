# Data Setup Instructions

The neural recommendation system requires a MovieLens dataset with the following structure:

## Required Data File

**File Name**: `movies_ratings_cleaned.csv`  
**Required Columns**:
- `userId` (or `user_id`, `User_ID`) - User identifier
- `movieId` (or `movie_id`, `Movie_ID`, `item_id`) - Movie identifier  
- `rating` - Rating value (typically 0.5-5.0)

## Data Sources

### Option 1: MovieLens Dataset (Recommended)
Download from: https://grouplens.org/datasets/movielens/

**Recommended datasets**:
- MovieLens 25M Dataset (25 million ratings)
- MovieLens 20M Dataset (20 million ratings)
- MovieLens Latest Small (100k ratings) - for testing

### Option 2: Alternative Datasets
- Amazon Product Reviews
- Netflix Prize Dataset
- Yelp Dataset
- Custom recommendation datasets

## Data Preparation

1. **Download the dataset**:
   ```bash
   # Example for MovieLens 25M
   wget http://files.grouplens.org/datasets/movielens/ml-25m.zip
   unzip ml-25m.zip
   ```

2. **Prepare the CSV file**:
   ```python
   import pandas as pd
   
   # Load ratings data
   ratings = pd.read_csv('ml-25m/ratings.csv')
   
   # Rename columns if needed
   ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
   
   # Remove timestamp if not needed
   ratings = ratings[['userId', 'movieId', 'rating']]
   
   # Save cleaned data
   ratings.to_csv('movies_ratings_cleaned.csv', index=False)
   ```

3. **Place the file**:
   - Copy `movies_ratings_cleaned.csv` to the project root directory
   - The file should be at the same level as `train.py`

## Data Requirements

- **Minimum size**: 100,000 ratings for meaningful training
- **Recommended size**: 1M+ ratings for better performance
- **Maximum tested**: 25M ratings (as implemented in this system)
- **Format**: CSV with headers
- **Encoding**: UTF-8

## Sample Data Structure

```csv
userId,movieId,rating
1,1,4.0
1,3,4.0
1,6,4.0
1,47,5.0
1,50,5.0
2,1,3.0
2,2,3.0
```

## Training Configuration

The system automatically:
- Samples 150,000 rows by default (configurable)
- Encodes user and movie IDs using LabelEncoder
- Splits data into 80% training, 20% testing
- Handles missing values and data validation

## File Size Considerations

- Large datasets (>100MB) are excluded from Git
- Use Git LFS for datasets >100MB if needed
- Consider data sampling for development/testing

## Troubleshooting

### Common Issues:
1. **File not found**: Ensure CSV is in root directory
2. **Column names**: System auto-detects variations
3. **Memory issues**: Reduce sample size in configuration
4. **Encoding errors**: Ensure UTF-8 encoding

### Support:
- Check logs during training for data validation
- Use `simple_inference.py` to test data loading
- Verify data format with pandas: `pd.read_csv('movies_ratings_cleaned.csv').info()`