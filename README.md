# Reproduction of a Hepatitis Classifier Study
## Reasoning behind reproduction
I'm planning on expanding my expertise and utility in a lab setting especially in a clinical environment, and I wanted a project that sits on the intersection of that goal and the coding skills that I'm trying to build now. Whilst searching for potential topics to touch up on, I often found medical data to be messy, high-stakes, and full of information that dictates the right and wrong calls. How are you able to determine and make judgment when data is missing? These aren't technical questions but rather ones posed in clinical environments daily.

This project was my way of practicing and preparing for both sides at once.

Rather than starting from scratch, I chose a published paper that resonated with my own ideals and tried to replicate their experiment myself, using tools I was still learning at the time (Python, pandas, scikit-learn). The paper:

> Rosly, R., Makhtar, M., Awang, M.K., Awang, M.I., Abdul Rahman, M.N. (2018).
> *Analyzing performance of classifiers for medical datasets.* International
> Journal of Engineering & Technology, 7(2.15), 136-138.

And the dataset we both used:

>Hepatitis dataset donated by G. Gong (Carnegie Mellon University) via
>Bojan Cestnik, Jozef Stefan Institute, Yugoslavia, November 1988.
>UCI Machine Learning Repository.
>https://archive.ics.uci.edu/dataset/46/hepatitis

They compared five classifiers and three ensemble methods on the UCI Hepatitis dataset using WEKA. Their comparison was a great foundation to
learn from, and this project is my attempt to reproduce it in Python, then extend it with a few pieces I was curious about.

## What "reproduction" meant for me

I rebuilt their pipeline from scratch:
- Same five classifier types (Naive Bayes, decision tree, k-NN, SVM, neural net)
- Same three ensemble strategies (bagging, boosting, stacking)
- Same 10-fold cross-validation setup

Gathering my numbers from the same dataset as theirs, using a completely different toolchain (scrikit-learn instead of WEKA), was just to test if I knew I understood their methodology or not.

### Handling missing data two ways

The dataset held a lot of missing values, for example one of the variables PROTIME was missing in 67 out of 155 patients. My original approach was to drop any incomplete row that was present in the dataset. By doing that, I also implemented an imputed version that keeps every patient in order to create a comparison between the two. It highlighted the significance in just how one single factor in preprocessing can move your results around, something I expected to happen in clinical research just as it occurred here.

### Going beyond accuracy

This dataset is imbalanced (123 patients survived, 32 didn't). I learned that accuracy alone can hide a model that's bad at catching the minority class, which in this scenario would be the patients missing variables. Knowing this, I added precision, recall, F1, and ROC-AUC for every model. Naive Bayes turned out to be the clearest example: worst accuracy between the five models, but the best recall of the DIE classification. That trade-off would've been invisible if I've only sought accuracy, and is the kind of thing I wanted to get comfortable reasoning before looking at real patient data in the future.

### Testing whether "better" is better

Ensembling gave a small accuracy bump in some cases, so I added a paired significance test across the cross-validation folds to check whether the increase was reliable or just null. This was something that was new to me going into the project, so seeing it actually occur is a lesson for me towards the future when I analyze data.

## Project structure

- data/ raw + cleaned datasets (dropped-rows and imputed versions)
- src/ preprocessing, single classifiers, ensembles
- results/ output tables and charts from each script
- results/comparison.md write-up comparing my numbers to the paper's

## Running it

- pip install -r requirements.txt
- python3 src/preprocess.py
- python3 src/single_classifiers.py
- python3 src/ensembles.py

## Documentation

To see results please consider looking [here](https://github.com/hudsonw2/hepatitis-data/blob/main/results/comparison.md).

