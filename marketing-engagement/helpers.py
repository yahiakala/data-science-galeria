"""Helper functions for Term Deposit Analysis."""
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_context('talk')


def visualize_model_performance(x_train, x_test, y_train, y_test,
                                model, threshold=0.5):
    """Summarize and visualize (binary classifier) model performance."""
    y_train_predp = model.predict_proba(x_train)[:, 1]
    y_test_predp = model.predict_proba(x_test)[:, 1]

    y_train_pred = (y_train_predp >= threshold).astype(int)
    y_test_pred = (y_test_predp >= threshold).astype(int)

    # Calculate accuracy, precision, recall, AUC.
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    prec_train = precision_score(y_train, y_train_pred)
    prec_test = precision_score(y_test, y_test_pred)
    rec_train = recall_score(y_train, y_train_pred)
    rec_test = recall_score(y_test, y_test_pred)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_predp)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_predp)
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    fig, ax = plt.subplots(2, 1, figsize=(12, 14))
    # Create a DataFrame (easier to plot).
    scores_df = pd.DataFrame(
        data=np.array([[acc_train, acc_test],
                       [prec_train, prec_test],
                       [rec_train, rec_test],
                       [auc_train, auc_test]]),
        columns=['Training Set', 'Test Set'],
        index=['Accuracy', 'Precision', 'Recall', 'AUC']
    )
    scores_df.plot(kind='bar', ax=ax[0])
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Score [-]')
    ax[0].set_title('Binary Classifier Performance')
    ax[0].set_ylim([0, 1.2])
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(0)

    for p in ax[0].patches:
        ax[0].annotate(
            str(p.get_height().round(2)),
            (p.get_x() + p.get_width()/2., p.get_height() * 1.005),
            ha='center', va='center', xytext=(0, 10),
            textcoords='offset points'
        )
    # Plot the ROC Curve.
    ax[1].plot(fpr_train, tpr_train, color='navy', label='Training Set')
    ax[1].plot(fpr_test, tpr_test, color='darkorange',
               label='Test Set')
    ax[1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax[1].grid()
    ax[1].set_xlim([0., 1.])
    ax[1].set_ylim([0., 1.05])
    ax[1].set_xlabel('False Positive Rate [-]')
    ax[1].set_ylabel('True Positive Rate [-]')
    ax[1].set_title('ROC Curve')
    ax[1].legend(loc='lower right')

    return fig, ax


def onehot_encode(dfin, onehot_cols):
    """One-hot encode selected columns."""
    df = dfin.copy()
    onehot_features = []
    for col in onehot_cols:
        # Create more cols for one-hot encoded features.
        encoded_df = pd.get_dummies(dfin[col])
        # Rename columns to add combinations
        encoded_df.columns = [col.replace(' ', '.') + '.' + x
                              for x in encoded_df.columns]
        onehot_features += list(encoded_df.columns)
        df = pd.concat([df, encoded_df], axis=1)
    return df[onehot_features]


def encode_features(dfin):
    """Encode features in a DataFrame and return final DF."""
    df = dfin.copy()
    # Find the non-numerical (categorical) columns.
    cnt_ft = list(df.describe().columns)

    cat_cols = [col for col in df.columns if col not in cnt_ft]

    # Find the columns to label encode (binary classes).
    label_cols = [col for col in cat_cols if
                  len(df[col].unique()) <= 2]

    # Find he columns to one-hot encode.
    onehot_cols = [col for col in cat_cols if col not in label_cols]

    # One-hot encode the columns.
    df2 = onehot_encode(dfin, onehot_cols)

    # Label encode the columns.
    label_features = [col + '_label' for col in label_cols]
    for col in label_cols:
        firstval = df[col].iloc[0]
        df[col + '_label'] = df[col].apply(lambda x: 1
                                           if x == firstval else 0)

    # Put it all together.
    keep_features = cnt_ft + label_features
    df_model = pd.concat([df[keep_features], df2], axis=1)
    return df_model


def get_ratio(input_df, conversion_col):
    """Calculate the conversion rate in a DataFrame."""
    return input_df[conversion_col].sum() / len(input_df)


def binary_process(input_df, binary_col):
    """Process conversion data."""
    output_df = input_df.copy()
    output_df[binary_col] = output_df[binary_col].str.lower()
    return output_df[binary_col].apply(lambda x: 1 if x == 'yes'
                                       else 0)


def ratio_segmented_1(input_df, conversion_col, seg_col):
    """Get the conversion rate on a 1D segmented dataset."""
    n1 = input_df.groupby(by=seg_col)[conversion_col].sum()
    n2 = input_df.groupby(by=seg_col)[conversion_col].count()
    return n1 / n2


def ratio_segmented_2(input_df, conversion_col, seg_cols):
    """Get the conversion rate on a 2D Segmented Dataset."""
    n1 = (input_df.groupby(seg_cols)[conversion_col]
          .sum().unstack(seg_cols[1]).fillna(0))
    n2 = input_df.groupby(seg_cols[0])[conversion_col].count()
    return n1.divide(n2, axis=0)


def ratio_segmented(input_df, ratio_col, seg_cols):
    """Get the conversion rate on a segmented dataset."""
    df1 = input_df.groupby(seg_cols)[ratio_col].sum()
    df2 = input_df.groupby(seg_cols)[ratio_col].count()
    return df1 / df2


def plot_ratios_and_bins(input_df, conversion_col, seg_col):
    """Plot conversion rates versus segmentation bins."""
    fig, ax = plt.subplots(figsize=(10, 7))
    convs = ratio_segmented_1(input_df, conversion_col, seg_col)
    ax.plot(convs.index, convs.values, color='blue')
    ax2 = ax.twinx()
    convc = input_df.groupby(seg_col)[conversion_col].count()
    ax2.plot(convc.index, convc.values, color='red')
    ax2.set_ylabel('Number of Data Points in Group', color='red')
    ax.set_ylabel('Conversion Rate (ratio)', color='blue')
    ax.set_xlabel(seg_col)
    return fig, ax


def bar_ratios_and_bins(input_df, ratio_col, seg_col):
    """Plot binary KPI ratio vs segmentation in a bar."""
    fig, ax = plt.subplots(2, 1, figsize=(16, 15))
    convs = ratio_segmented_1(input_df, ratio_col, seg_col)
    ax[0].bar(convs.index, convs.values, color='skyblue')
    ax[0].set_ylabel(ratio_col + ' Rate [-]')
    # convc = input_df.groupby(seg_col)[ratio_col].count()
    # ax[1].bar(convc.index, convc.values, color='red')
    dummy_df = input_df.copy()
    dummy_df['dum1'] = dummy_df[ratio_col].apply(lambda x: 'Yes' if x == 1
                                                 else 'No')
    dummy_df['KPI'] = dummy_df['dum1']
    convc2 = pd.pivot_table(dummy_df, values='dum1', columns='KPI',
                            index=seg_col,
                            aggfunc=len).fillna(0.0)
    convc2.plot(ax=ax[1], kind='bar', stacked='True')
    for tick1, tick2 in zip(ax[0].get_xticklabels(),
                            ax[1].get_xticklabels()):
        tick1.set_rotation(30)
        tick2.set_rotation(30)
    ax[1].set_ylabel('Data Points')
    ax[1].set_xlabel(seg_col)
    return fig, ax


def stacked_bar_ratios(input_df, conversion_col, seg_cols):
    """Plot a stacked bar with a segmented population."""
    df = ratio_segmented_2(input_df, conversion_col,
                           seg_cols)
    fig, ax = plt.subplots(2, 1, figsize=(16, 15))
    df.plot(ax=ax[0], kind='bar', stacked='True')
    ax[0].set_title('KPI Rates by ' + str(seg_cols[0]) + ' and ' +
                    str(seg_cols[1]))
    ax[0].set_ylabel(conversion_col + ' Rate [-]')
    ax[0].set_xlabel('')
    convc = input_df.groupby(seg_cols)[conversion_col].count().unstack(
        seg_cols[1]).fillna(0)
    convc.plot(ax=ax[1], kind='bar', stacked='True')
    ax[1].set_ylabel('Data Points')
    ax[1].set_xlabel(seg_cols[0])
    for tick1, tick2 in zip(ax[0].get_xticklabels(),
                            ax[1].get_xticklabels()):
        tick1.set_rotation(30)
        tick2.set_rotation(30)
    return fig, ax


def segment_scatter_2(input_df, col1, col2, scol1, scol2):
    """Plot a scatter of segmented data."""
    fig, ax = plt.subplots(figsize=(10, 7))
    inds = []
    for i in ['High', 'Low']:
        for j in ['High', 'Low']:
            inds.append(
                np.where(
                    (input_df[scol1] == i) & (input_df[scol2] == j)
                )[0]
            )
    clrs = ['red', 'blue', 'orange', 'green']
    for ind, clr in zip(inds, clrs):
        ax.scatter(
            input_df[col1].iloc[ind],
            input_df[col2].iloc[ind], c=clr, s=2
        )
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    return fig, ax
