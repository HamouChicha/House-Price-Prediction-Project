import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


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


# function to plot the heat map of th correlated features above a certain threshold
def heat_map(X, numerical_features, threshold=0.5):
    # Calculate the correlation matrix
    correlation_matrix = X[numerical_features].corr()

    # Create a mask to filter out correlations below the threshold
    mask = correlation_matrix.abs() > threshold

    # Apply the mask to the correlation matrix
    filtered_correlation_matrix = correlation_matrix[mask].dropna(axis=0, how='all').dropna(axis=1, how='all')

    # Set up the matplotlib figure
    plt.figure(figsize=(20, 20))

    # Create a heatmap using Seaborn
    sns.heatmap(filtered_correlation_matrix, 
                annot=True,             # Annotate cells with the correlation coefficient
                cmap='RdBu',           # Color map
                center=0,              # Center the color map at 0
                square=False,           # Make cells square
                cbar_kws={"shrink": .9})  # Color bar size

    # Set the titles and labels
    plt.title('Filtered Correlation Heatmap', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # Show the plot
    plt.show()

# function to visualize the lowest feature variances
def plot_low_variance(X, num_cols):
    variances = np.var(X, axis=0)  # Calculate variance for each feature
    feature_names = X.columns  # Assuming X is a DataFrame

    # Create a DataFrame for better handling
    variance_df = pd.DataFrame({'Feature': feature_names, 'Variance': variances})

    # Sort the DataFrame by variance
    variance_df = variance_df.sort_values(by='Variance', ascending=False)
    # Plotting
    variance_df.nsmallest(num_cols, 'Variance').plot(kind='barh', x='Feature', y='Variance', figsize=(10, 6), color='teal')
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
    feature_df.nlargest(num_cols, 'Score').plot(kind='barh', x='Feature', y='Score', figsize=(10, 6), color='teal')
    plt.title(f'Top {num_cols} Feature Scores')
    plt.xlabel('Score')
    plt.ylabel('Features')
    plt.show()


# function to evaluate the model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    print("---------------------- Evaluating on training data ...")
    y_pred = model.predict(X_train)
    print(f"Train MSE: {mean_squared_error(y_train, y_pred):.4f}\nTrain R2: {r2_score(y_train, y_pred):.4f}")

    print("---------------------- Evaluating on testing data ...")
    y_pred = model.predict(X_test)
    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f},\nTest R2: {r2_score(y_test, y_pred):.4f}")

    
    

