import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# function to plot the box plot of features
def box_plot(X, features, n):
    # Determine the number of rows needed for the subplots
    n_rows = (len(features) + n - 1) // n  # This ensures we round up

    # Create subplots
    fig, axes = plt.subplots(n_rows, n, figsize=(n * 5, n_rows * 4))

    # Flatten axes array for easy iteration
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, feature in enumerate(features):
        sns.boxplot(data=X, y=feature, ax=axes[i])
        axes[i].set_title(f'Box Plot of {feature}')
        axes[i].set_xlabel('')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# function to plot the interactive heat map of features
def heat_map(X, numerical_features):
    # Calculate the correlation matrix
    correlation_matrix = X[numerical_features].corr()

    # Create an interactive heatmap using Plotly
    fig = px.imshow(correlation_matrix,
                    color_continuous_scale='Viridis',
                    title='Correlation Heatmap',
                    labels=dict(x='Features', y='Features', color='Correlation'),
                    aspect='auto')

    # Show the figure
    fig.show()


# function to visualize the lowest feature variances
def plot_low_variance(X, num_cols):
    variances = np.var(X, axis=0)  # Calculate variance for each feature
    feature_names = X.columns  # Assuming X is a DataFrame

    # Create a DataFrame for better handling
    variance_df = pd.DataFrame({'Feature': feature_names, 'Variance': variances})

    # Sort the DataFrame by variance
    variance_df = variance_df.sort_values(by='Variance', ascending=False)
    # Plotting
    top_n = num_cols # Number of top features to display
    feature_df.nlargest(top_n, 'Variance').plot(kind='barh', x='Feature', y='Variance', figsize=(10, 6), color='teal')
    plt.title(f'Lowest {num_cols} Feature Variances')
    plt.xlabel('Variance')
    plt.ylabel('Features')
    plt.show()


# function to plot the top k features based on their scores 
def plot_k_best(selector, feature_names, num_cols):
    # Assuming feature_scores is a list or array of scores and feature_names is a corresponding list of feature names
    feature_scores = selector.scores_

    # Create a DataFrame for better handling
    feature_df = pd.DataFrame({'Feature': feature_names, 'Score': feature_scores})

    # Sort the DataFrame by score
    feature_df = feature_df.sort_values(by='Score', ascending=False)

    # Plotting
    top_n = num_cols # Number of top features to display
    feature_df.nlargest(top_n, 'Score').plot(kind='barh', x='Feature', y='Score', figsize=(10, 6), color='teal')
    plt.title('Top N Feature Scores')
    plt.xlabel('Score')
    plt.ylabel('Features')
    plt.show()