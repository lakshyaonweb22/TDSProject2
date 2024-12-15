import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai

# Set API key and base URL for the proxy
openai.api_key = os.getenv("AIPROXY_TOKEN")
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"  # Updated base URL


# Function to load the dataset
def load_dataset(file_path):
    try:
        # Attempt reading with default UTF-8 encoding
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with UTF-8! Shape: {df.shape}")
        return df
    except UnicodeDecodeError:
        try:
            # Fallback to ISO-8859-1 encoding
            df = pd.read_csv(file_path, encoding="ISO-8859-1")
            print(f"Dataset loaded successfully with ISO-8859-1! Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)


# Function to clean the dataset
def preprocess_data(df):
    print("Preprocessing dataset...")
    missing_summary = df.isnull().sum()
    print(f"Missing values:\n{missing_summary}\n")
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna("Unknown")  # For non-numeric columns
    return df


# Function to generate summary statistics
def generate_summary(df):
    print("Generating summary statistics...")
    summary = df.describe(include="all")
    print(summary)
    return summary


# Function to visualize data
def create_visualizations(df, output_dir):
    print("Creating visualizations...")
    try:
        numeric_data = df.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr = numeric_data.corr()
            # plt.figure(figsize=(5.12, 5.12), dpi=100)
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")
            heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path)
            print(f"Saved {heatmap_path}")

        if not numeric_data.empty:
            # plt.figure(figsize=(5.12, 5.12), dpi=100)
            plt.figure(figsize=(10, 8))
            numeric_data.hist(bins=30, figsize=(5.12, 5.12))
            plt.tight_layout()
            distributions_path = os.path.join(output_dir, "distributions.png")
            plt.savefig(distributions_path)
            print(f"Saved {distributions_path}")
    except Exception as e:
        print(f"Error creating visualizations: {e}")


# Function to narrate the analysis using LLM
def narrate_story(df):
    print("Narrating the story using LLM...")
    columns = df.columns.tolist()
    sample_data = df.head(5).to_dict()
    summary = f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. Columns are: {columns}."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Supported model
            messages=[
                {"role": "system", "content": "You are a data analyst."},
                {
                    "role": "user",
                    "content": f"Here is a summary of the dataset: {summary}. The first five rows are: {sample_data}. Write a story about this dataset.",
                },
            ],
        )
        story = response["choices"][0]["message"]["content"]
        print("Story generated successfully!")
        return story
    except Exception as e:
        print(f"Error communicating with the LLM: {e}")
        sys.exit(1)


# Main script
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    # Get the folder name based on the CSV file name
    file_name = os.path.basename(file_path)
    folder_name = os.path.splitext(file_name)[0]

    # Create a directory for the dataset
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

    # Load, process, and analyze the dataset
    df = load_dataset(file_path)
    df = preprocess_data(df)
    generate_summary(df)

    # Create visualizations and save in the output directory
    create_visualizations(df, output_dir)

    # Generate story and save to README
    story = narrate_story(df)
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Automated Analysis\n\n")
        f.write("## Narration\n")
        f.write(story)
        f.write("\n\n## Visualizations\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)\n")
        f.write("![Distributions](distributions.png)\n")

    print(f"Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
