import numpy as np
import pandas
import tf_access


def one_hot_binary_reconstruct(training_data, testing_data):
    print("training data:")
    print(training_data)

    """
    ROLE_DEPTNAME reconstruction (1st best predictor)
    """
    # Create one-hot binary inputs columns to re-construct training and testing data
    role_family_one_hot_tr = pandas.get_dummies(training_data.pop("ROLE_DEPTNAME"),
                                                prefix='ROLE_DEPTNAME')  # 1) good one
    role_family_one_hot_te = pandas.get_dummies(testing_data.pop("ROLE_DEPTNAME"), prefix='ROLE_DEPTNAME')

    # align the columns and fill null values with 0s
    role_family_one_hot_tr, role_family_one_hot_te = role_family_one_hot_tr.align(role_family_one_hot_te, join='outer',
                                                                                  axis=1)
    role_family_one_hot_tr = role_family_one_hot_tr.fillna(0)
    role_family_one_hot_te = role_family_one_hot_te.fillna(0)
    # Re-construct into training data
    training_data = training_data.join(role_family_one_hot_tr)
    testing_data = testing_data.join(role_family_one_hot_te)

    """
    ROLE_CODE reconstruction (2nd best predictor)
    """
    # Create one-hot binary inputs columns to re-construct training and testing data
    role_code_one_hot_tr = pandas.get_dummies(training_data.pop("ROLE_CODE"), prefix='ROLE_CODE')
    role_code_one_hot_te = pandas.get_dummies(testing_data.pop("ROLE_CODE"), prefix='ROLE_CODE')
    # align the columns and fill null values with 0s
    role_code_one_hot_tr, role_code_one_hot_te = role_code_one_hot_tr.align(role_code_one_hot_te, join='outer',
                                                                            axis=1)
    role_code_one_hot_tr = role_code_one_hot_tr.fillna(0)
    role_code_one_hot_te = role_code_one_hot_te.fillna(0)
    # Re-construct into training data
    training_data = training_data.join(role_code_one_hot_tr)
    testing_data = testing_data.join(role_code_one_hot_te)
    """
    ROLE_TITLE reconstruction (3rd best predictor)
    """
    # Create one-hot binary inputs columns to re-construct training and testing data
    role_title_one_hot_tr = pandas.get_dummies(training_data.pop("ROLE_TITLE"), prefix='ROLE_TITLE')
    role_title_one_hot_te = pandas.get_dummies(testing_data.pop("ROLE_TITLE"), prefix='ROLE_TITLE')
    # align the columns and fill null values with 0s
    role_title_one_hot_tr, role_title_one_hot_te = role_title_one_hot_tr.align(role_title_one_hot_te, join='outer',
                                                                               axis=1)
    role_title_one_hot_tr = role_title_one_hot_tr.fillna(0)
    role_title_one_hot_te = role_title_one_hot_te.fillna(0)
    # Re-construct into training data
    training_data = training_data.join(role_title_one_hot_tr)
    testing_data = testing_data.join(role_title_one_hot_te)

    """
    ROLE_ROLLUP_2 reconstruction (4th best predictor)
    """
    # Create one-hot binary inputs columns to re-construct training and testing data
    role_title_one_hot_tr = pandas.get_dummies(training_data.pop("ROLE_ROLLUP_2"), prefix='ROLE_ROLLUP_2')
    role_title_one_hot_te = pandas.get_dummies(testing_data.pop("ROLE_ROLLUP_2"), prefix='ROLE_ROLLUP_2')
    # align the columns and fill null values with 0s
    role_title_one_hot_tr, role_title_one_hot_te = role_title_one_hot_tr.align(role_title_one_hot_te, join='outer',
                                                                               axis=1)
    role_title_one_hot_tr = role_title_one_hot_tr.fillna(0)
    role_title_one_hot_te = role_title_one_hot_te.fillna(0)
    # Re-construct into training data
    training_data = training_data.join(role_title_one_hot_tr)
    testing_data = testing_data.join(role_title_one_hot_te)

    return training_data, testing_data


def repeat_action_zero_columns(training_data):
    """
    NOTE: Training data is heavily skewed with data where ACTION = 1
    --> ACTION column value analysis: 1s: 30872, 0s: 1897, total:32769
    --> We'll repeat the data instances when ACTION = 0 to get a 50-50 ratio
    """
    zero_rows = training_data[training_data['ACTION'] == 0]  # Find rows where ACTION == 0
    zero_rows = zero_rows.loc[zero_rows.index.repeat(15)]  # Repeat all those rows 15 times
    zero_rows = zero_rows.sample(frac=1).reset_index(drop=True)  # shuffle data
    # append rows and shuffle again
    training_data = pandas.concat([training_data, zero_rows], ignore_index=True, sort=False) \
        .sample(frac=1).reset_index(drop=True)
    return training_data


if __name__ == "__main__":
    training_data = pandas.read_csv("./train.csv")  # Load training data
    testing_data = pandas.read_csv("./test.csv")  # Load test data

    # Select ground truth & desired features for training and testing data
    training_data = training_data[["ACTION", "ROLE_DEPTNAME", "ROLE_CODE", "ROLE_TITLE", "ROLE_ROLLUP_2"]]
    testing_data = testing_data[["id", "ROLE_DEPTNAME", "ROLE_CODE", "ROLE_TITLE", "ROLE_ROLLUP_2"]]

    # PRE-PROCESS #1: Convert some desired features to one-hot binary
    training_data, testing_data = one_hot_binary_reconstruct(training_data, testing_data)
    # print("After one-hot:")
    # print(training_data)
    # PRE-PROCESS #2: Pad training data with more instances of ACTION = 0
    training_data = repeat_action_zero_columns(training_data)
    # print("After padding:")
    # print(training_data)
    # Train with a model
    y_test = tf_access.train(training_data, testing_data)

    # Write to CSV file
    ids = testing_data.id.to_numpy()
    actions = y_test

    df = pandas.DataFrame({"Id": ids, "Action": actions})
    df.to_csv("prediction.csv", index=False)
