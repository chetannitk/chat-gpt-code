import pandas as pd
import re

def filter_dataframe_by_regex(df, column_name, regex_patterns, logic='OR', target_column=None):
    """
    Filter a DataFrame based on regex patterns applied to a specific column.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        column_name (str): The name of the column to filter.
        regex_patterns (list): A list of lists containing regex patterns to filter the column.
                              Each inner list represents patterns combined with AND logic.
                              Different inner lists will be combined with OR logic.
        logic (str, optional): The logic to combine patterns within each inner list.
                               Valid options are 'AND' and 'OR'. Default is 'OR'.
        target_column (str, optional): The name of the column where the matched parts will be stored.
                                       If not provided, the matched parts will be stored in a new column
                                       named "Matched_Part".

    Returns:
        pandas.DataFrame: A new DataFrame containing rows that match the specified regex patterns.
                         The DataFrame includes an additional column with matched parts (or the entire
                         matched string if the target_column is not specified).
    """
    if logic not in ['AND', 'OR']:
        raise ValueError("Invalid logic parameter. Use 'AND' or 'OR'.")

    filtered_rows = pd.Series(False, index=df.index)
    matched_parts = pd.Series("", index=df.index)

    for pattern_group in regex_patterns:
        if logic == 'OR':
            regex_condition = df[column_name].str.contains('|'.join(pattern_group), na=False, flags=re.IGNORECASE)
        else:  # logic == 'AND'
            and_conditions = [df[column_name].str.contains(pattern, na=False, flags=re.IGNORECASE) for pattern in pattern_group]
            regex_condition = pd.Series(True, index=df.index)
            for condition in and_conditions:
                regex_condition = regex_condition & condition

        filtered_rows = filtered_rows | regex_condition

        # Update matched parts for the current group
        matches = df.loc[regex_condition, column_name].str.findall('({})'.format('|'.join(pattern_group)), re.IGNORECASE)
        matched_parts.loc[matches.index] = matches

    if target_column is None:
        target_column = "Matched_Part"

    filtered_df = df[filtered_rows].copy()
    filtered_df[target_column] = matched_parts

    return filtered_df

# Example usage:
if __name__ == "__main__":
    data = {
        "Name": ["John Doe", "Alice Smith", "Bob Johnson", "Eve", "Michael Jackson"],
        "Age": [30, 25, 40, 22, 50],
        "City": ["New York", "Los Angeles", "Chicago", "Boston", "Las Vegas"]
    }

    df = pd.DataFrame(data)

    regex_patterns = [
        ["^J|Doe", "son$"],       # Match names that start with 'J' or end with 'son'
        ["^A", "Los Angeles"] # Match names that start with 'A' and city is 'Los Angeles'
    ]

    filtered_df = filter_dataframe_by_regex(df, "Name", regex_patterns, logic='OR', target_column="Matched_Name")
    print(filtered_df)
