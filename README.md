# Autolysis: Automated Dataset Analysis Tool

Autolysis is a Python-based tool for automating dataset analysis, including preprocessing, generating summary statistics, visualizing data, and narrating insights using a language model.

## Features

- **Dataset Loading**: Automatically handles encoding errors (supports UTF-8 and ISO-8859-1).
- **Data Cleaning**: Fills missing values with the mean for numeric columns and "Unknown" for non-numeric columns.
- **Summary Statistics**: Provides a comprehensive statistical dataset summary.
- **Data Visualization**:
  - Correlation Heatmap
  - Histograms of Numeric Columns
- **LLM-Based Narration**: Leverages GPT to generate a story based on dataset insights.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/lakshyaonweb22/autolysis.git
   cd autolysis
