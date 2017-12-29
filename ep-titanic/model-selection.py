import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#Fixme. Please use the new model_selection class against the old an deprecated cross_validation and grid_search!
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# References:
# https://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html
# https://www.datacamp.com/community/tutorials/exploratory-data-analysis-python
# http://scikit-learn.org/stable/modules/cross_validation.html
# http://scikit-learn.org/stable/modules/grid_search.html#grid-search


def get_combined_data():
    # reading train data
    train = pd.read_csv('train.csv')

    # reading test data
    test = pd.read_csv('test.csv')

    # extracting and then removing the targets from the training data
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)

    return combined


def explore(combined):
    print("\n***Let'slook at the shape of our data frame")
    print(combined.shape)

    print("\n***Let's see how the first entries look like")
    print(combined.head())

    print("\n***Let's check the data type of each variable")
    print(combined.dtypes)

    print("\n***Let's check basic stats of numeric variables")
    print(combined.describe())

    # Fixme: add more!


def prepare(combined):
    def status(feature):
        print('Processing', feature, ': ok')

    def get_titles(combined):
        # we extract the title from each name
        combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

        # a map of more aggregated titles
        Title_Dictionary = {
            "Capt": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Jonkheer": "Royalty",
            "Don": "Royalty",
            "Sir": "Royalty",
            "Dr": "Officer",
            "Rev": "Officer",
            "the Countess": "Royalty",
            "Dona": "Royalty",
            "Mme": "Mrs",
            "Mlle": "Miss",
            "Ms": "Mrs",
            "Mr": "Mr",
            "Mrs": "Mrs",
            "Miss": "Miss",
            "Master": "Master",
            "Lady": "Royalty"

        }

        # we map each title
        combined['Title'] = combined.Title.map(Title_Dictionary)
        status('Title')
        return combined

    def process_age(combined):
        grouped_train = combined.head(891).groupby(['Sex', 'Pclass', 'Title'])
        grouped_median_train = grouped_train.median()

        grouped_test = combined.iloc[891:].groupby(['Sex', 'Pclass', 'Title'])
        grouped_median_test = grouped_test.median()

        # a function that fills the missing values of the Age variable

        def fillAges(row, grouped_median):
            if row['Sex'] == 'female' and row['Pclass'] == 1:
                if row['Title'] == 'Miss':
                    return grouped_median.loc['female', 1, 'Miss']['Age']
                elif row['Title'] == 'Mrs':
                    return grouped_median.loc['female', 1, 'Mrs']['Age']
                elif row['Title'] == 'Officer':
                    return grouped_median.loc['female', 1, 'Officer']['Age']
                elif row['Title'] == 'Royalty':
                    return grouped_median.loc['female', 1, 'Royalty']['Age']

            elif row['Sex'] == 'female' and row['Pclass'] == 2:
                if row['Title'] == 'Miss':
                    return grouped_median.loc['female', 2, 'Miss']['Age']
                elif row['Title'] == 'Mrs':
                    return grouped_median.loc['female', 2, 'Mrs']['Age']

            elif row['Sex'] == 'female' and row['Pclass'] == 3:
                if row['Title'] == 'Miss':
                    return grouped_median.loc['female', 3, 'Miss']['Age']
                elif row['Title'] == 'Mrs':
                    return grouped_median.loc['female', 3, 'Mrs']['Age']

            elif row['Sex'] == 'male' and row['Pclass'] == 1:
                if row['Title'] == 'Master':
                    return grouped_median.loc['male', 1, 'Master']['Age']
                elif row['Title'] == 'Mr':
                    return grouped_median.loc['male', 1, 'Mr']['Age']
                elif row['Title'] == 'Officer':
                    return grouped_median.loc['male', 1, 'Officer']['Age']
                elif row['Title'] == 'Royalty':
                    return grouped_median.loc['male', 1, 'Royalty']['Age']

            elif row['Sex'] == 'male' and row['Pclass'] == 2:
                if row['Title'] == 'Master':
                    return grouped_median.loc['male', 2, 'Master']['Age']
                elif row['Title'] == 'Mr':
                    return grouped_median.loc['male', 2, 'Mr']['Age']
                elif row['Title'] == 'Officer':
                    return grouped_median.loc['male', 2, 'Officer']['Age']

            elif row['Sex'] == 'male' and row['Pclass'] == 3:
                if row['Title'] == 'Master':
                    return grouped_median.loc['male', 3, 'Master']['Age']
                elif row['Title'] == 'Mr':
                    return grouped_median.loc['male', 3, 'Mr']['Age']

        #Fixme, see the warnings
        combined.head(891).Age = combined.head(891).apply(
            lambda r: fillAges(r, grouped_median_train) if np.isnan(r['Age'])
            else r['Age'], axis=1)

        combined.iloc[891:].Age = combined.iloc[891:].apply(
            lambda r: fillAges(r, grouped_median_test) if np.isnan(r['Age'])
            else r['Age'], axis=1)

        status('age')
        return combined

    def process_names(combined):

        # we clean the Name variable
        combined.drop('Name', axis=1, inplace=True)

        # encoding in dummy variable
        titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
        combined = pd.concat([combined, titles_dummies], axis=1)

        # removing the title variable
        combined.drop('Title', axis=1, inplace=True)

        status('names')
        return combined

    def process_fares(combined):
        # Fixme, see the warnings
        # there's one missing fare value - replacing it with the mean.
        combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace=True)
        combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace=True)

        status('fare')
        return combined

    def process_embarked(combined):

        # two missing embarked values - filling them with the most frequent one (S)
        combined.head(891).Embarked.fillna('S', inplace=True)
        combined.iloc[891:].Embarked.fillna('S', inplace=True)

        # dummy encoding
        embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
        combined = pd.concat([combined, embarked_dummies], axis=1)
        combined.drop('Embarked', axis=1, inplace=True)

        status('embarked')
        return combined

    def process_cabin(combined):

        # replacing missing cabins with U (for Uknown)
        combined.Cabin.fillna('U', inplace=True)

        # mapping each Cabin value with the cabin letter
        combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

        # dummy encoding ...
        cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')

        combined = pd.concat([combined, cabin_dummies], axis=1)

        combined.drop('Cabin', axis=1, inplace=True)

        status('cabin')
        return combined

    def process_sex(combined):
        # mapping string values to numerical one
        combined['Sex'] = combined['Sex'].map({'male': 1, 'female': 0})

        status('sex')
        return combined

    def process_pclass(combined):

        # encoding into 3 categories:
        pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

        # adding dummy variables
        combined = pd.concat([combined, pclass_dummies], axis=1)

        # removing "Pclass"

        combined.drop('Pclass', axis=1, inplace=True)

        status('pclass')
        return combined

    def process_family(combined):

        # introducing a new feature : the size of families (including the passenger)
        combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

        # introducing other features based on the family size
        combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
        combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
        combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

        status('family')
        return combined


    combined = get_titles(combined)
    combined = process_age(combined)
    combined = process_names(combined)
    combined = process_fares(combined)
    combined = process_embarked(combined)
    combined = process_cabin(combined)
    combined = process_sex(combined)
    combined = process_pclass(combined)
    #combined = process_ticket(combined)
    combined.drop('Ticket', inplace=True, axis=1)

    combined = process_family(combined)
    combined.drop('PassengerId', inplace=True, axis=1)
    print(combined.shape)
    return combined


def recover_train_test_target(combined):
    train0 = pd.read_csv('train.csv')

    targets = train0.Survived
    train = combined.head(891)
    test = combined.iloc[891:]

    return train, test, targets


def compute_score(clf, X, y, scoring='accuracy', defaul_cv = 5):
    ''' This function calculates the mean accuracy of the prediction of the classifier clf
    on the training set X and against the target values Y using a k fold cross validation.
    It returns the mean accuracy and std
    '''
    xval = cross_val_score(clf, X, y, cv=defaul_cv, scoring=scoring)
    return np.mean(xval), np.std(xval)


def svc_classify(train, targets):
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-5],
                         'C': [100, 1000, 10000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100]}]
    scores = ['precision', 'recall']
    X_train, X_test, y_train, y_test = train_test_split(
        train, targets, test_size=0.5, random_state=0)

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    # Best parameter for precision: {'C': 10000, 'gamma': 1e-05, 'kernel': 'rbf'}
    # Best parameter for recall:    {'C': 10000, 'gamma': 1e-05, 'kernel': 'rbf'}


def random_forest_classify(train, targets):
    parameter_grid = {
        'max_depth': [5, 6, 7],
        'n_estimators': [15, 16, 17, 18],
        'max_features': ['sqrt'],  #, 'auto', 'log2'],
        'min_samples_split': [5],
        'min_samples_leaf': [1],
        'bootstrap': [True, False],
    }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train, targets)
    model = grid_search

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

    mean, std = compute_score(model, train, targets, scoring='accuracy')
    print('Accuracy= %.2f +- %.2f' % (mean, 2*std))

'''
Best score: 0.8383838383838383
Best parameters: {'bootstrap': True, 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 11}
Accuracy= 0.82 +- 0.05

Best score: 0.8361391694725028
Best parameters: {'bootstrap': True, 'max_depth': 7, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 15}
Accuracy= 0.82 +- 0.03

Best score: 0.8338945005611672
Best parameters: {'bootstrap': False, 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 16}
Accuracy= 0.83 +- 0.03

'''
def main():
    # Load data
    combined = get_combined_data()

    # Data Exploration
    explore(combined)

    # Data Preparation
    combined = prepare(combined)

    train, test, targets = recover_train_test_target(combined)

    #svc_classify(train, targets)
    random_forest_classify(train, targets)


if __name__ == '__main__':
    main()
